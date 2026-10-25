"""Compare two `adaforge score` outputs (reference vs candidate).

Reports top-1 agreement and KL(reference || candidate) per position, computed
over the reference top-k (a candidate logit missing from its own top-k is
bounded by its smallest listed logit). Exact kernels give 100 % / 0.
"""
import argparse, math, statistics

def load(path):
    seqs, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            if cur: seqs.append(cur); cur = []
            continue
        head, *pairs = line.split()
        top = [(int(a), float(b)) for a, b in (p.split(":") for p in pairs)]
        cur.append((float(head), top))
    if cur: seqs.append(cur)
    return seqs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference"); p.add_argument("candidate")
    a = p.parse_args()
    ref, cand = load(a.reference), load(a.candidate)
    assert len(ref) == len(cand), "different number of sequences"
    agree = total = 0; kls = []
    for rs, cs in zip(ref, cand):
        assert len(rs) == len(cs)
        for (rz, rt), (cz, ct) in zip(rs, cs):
            total += 1
            agree += rt[0][0] == ct[0][0]
            cmap = dict(ct); floor = ct[-1][1]
            kl = 0.0
            for tid, rl in rt:
                pr = math.exp(rl - rz)
                kl += pr * ((rl - rz) - (cmap.get(tid, floor) - cz))
            kls.append(max(kl, 0.0))
    kls.sort()
    print(f"positions {total}  top-1 identique {agree/total:.2%}  "
          f"KL moyenne {statistics.mean(kls):.2e}  p99 {kls[int(0.99*(len(kls)-1))]:.2e}  max {kls[-1]:.2e}")

if __name__ == "__main__":
    main()
