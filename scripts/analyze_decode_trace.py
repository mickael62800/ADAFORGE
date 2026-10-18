"""Summarize GPU kernels and unobserved gaps between repeated decode anchors."""
import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import statistics


def union_ns(intervals):
    total = 0
    finish = None
    for start, end in sorted(intervals):
        if finish is None or start > finish:
            total += end - start
        else:
            total += max(0, end - finish)
        finish = max(end, finish) if finish is not None else end
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('database', type=Path)
    p.add_argument('--anchor', default='^greedy_final$')
    p.add_argument('--last-anchors', type=int, default=24)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.last_anchors < 3:
        p.error('last-anchors >= 3')
    with closing(sqlite3.connect(f'file:{args.database.resolve().as_posix()}?mode=ro', uri=True)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        names = dict(conn.execute('SELECT id,value FROM StringIds'))
        kernels = [(s, e, names[n]) for s, e, n in conn.execute(
            'SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')]
        anchors = [s for s, _, n in kernels if re.search(args.anchor, n)]
        if len(anchors) < args.last_anchors:
            raise RuntimeError(f'Only {len(anchors)} anchors found, need {args.last_anchors}')
        anchors = anchors[-args.last_anchors:]
        events = [(s, e) for s, e, _ in kernels]
        for table in ['CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET']:
            if table in tables:
                events.extend(conn.execute(f'SELECT start,end FROM {table}'))
    lo, hi = anchors[0], anchors[-1]
    grouped = defaultdict(list)
    for start, end, name in kernels:
        if start >= lo and end <= hi:
            grouped[name].append(end - start)
    steps = []
    for start, end in zip(anchors, anchors[1:]):
        covered = union_ns([(max(s, start), min(e, end)) for s, e in events if s < end and e > start])
        steps.append({'wall_ms': (end - start) / 1e6, 'tracked_gpu_ms': covered / 1e6,
                      'uncovered_ms': (end - start - covered) / 1e6})
    total_kernel = sum(sum(values) for values in grouped.values())
    top = [{'kernel': name, 'instances': len(values), 'sum_ms': sum(values) / 1e6,
            'median_us': statistics.median(values) / 1e3,
            'max_us': max(values) / 1e3, 'kernel_sum_percent': 100 * sum(values) / total_kernel}
           for name, values in grouped.items()]
    top.sort(key=lambda row: row['sum_ms'], reverse=True)
    report = {'database': str(args.database.resolve()), 'anchor_regex': args.anchor,
              'anchor_count': len(anchors), 'window_start_ns': lo, 'window_end_ns': hi,
              'step_count': len(steps), 'steps': steps, 'top_kernels': top,
              'median_step_ms': {key: statistics.median(row[key] for row in steps) for key in steps[0]},
              'max_anchor_gap_ms': max(row['wall_ms'] for row in steps),
              'notes': ['Anchor intervals are not HTTP or engine token-ready measurements.',
                        'Tracked GPU time is a union of recorded kernels/copies/memsets, without double counting.',
                        'Uncovered time has no causal attribution: host work, driver scheduling and untracked work may contribute.',
                        'Per-kernel percentages use summed kernel durations, not end-to-end wall time.']}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['step_count', 'median_step_ms', 'max_anchor_gap_ms']}, indent=2))


if __name__ == '__main__':
    main()
