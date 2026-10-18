"""Validate interval accounting and anchor slicing on a small synthetic trace."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from analyze_decode_trace import union_ns


class DecodeTraceAccounting(unittest.TestCase):
    def test_union_does_not_double_count_overlap(self):
        for intervals, expected in [([], 0), ([(1, 10), (2, 3)], 9),
                                    ([(0, 2), (2, 4)], 4),
                                    ([(5, 7), (1, 4), (3, 6)], 6)]:
            with self.subTest(intervals=intervals):
                self.assertEqual(union_ns(intervals), expected)

    def test_anchors_clip_crossing_kernels_and_merge_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / 'trace.sqlite'
            output = Path(directory) / 'report.json'
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute('CREATE TABLE StringIds (id INTEGER, value TEXT)')
                conn.executemany('INSERT INTO StringIds VALUES (?,?)', [(1, 'greedy_final'), (2, 'work')])
                conn.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)')
                conn.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)',
                                 [(0, 10, 1), (100, 110, 1), (200, 210, 1),
                                  (20, 80, 2), (90, 120, 2), (150, 180, 2)])
                conn.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER)')
                conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (85,95)')
            subprocess.run([sys.executable, str(Path(__file__).with_name('analyze_decode_trace.py')),
                            str(db), '--last-anchors', '3', '--output', str(output)],
                           check=True, capture_output=True)
            result = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(result['step_count'], 2)
        self.assertAlmostEqual(result['steps'][0]['tracked_gpu_ms'], 85 / 1e6)
        self.assertAlmostEqual(result['steps'][1]['tracked_gpu_ms'], 50 / 1e6)
        self.assertAlmostEqual(result['steps'][0]['uncovered_ms'], 15 / 1e6)
        self.assertAlmostEqual(result['steps'][1]['uncovered_ms'], 50 / 1e6)


if __name__ == '__main__':
    unittest.main()
