"""Grade the runs with Steer2Edit's own extract_answer / verify_answer, unmodified.

    cd efficient_reasoning && python reencode/evaluate_reencode.py      (needs math_verify only; no GPU)
Writes reencode_results/eval/math100.jsonl (one graded record per generation) and prints the readout: per (arm, alpha)
accuracy, mean reasoning length (their metric: tokens of the stripped response) and cap hits, plus the paired
add_re vs add comparison at each alpha (one-arm-only correct counts with a sign test; mean length difference and
geometric-mean length ratio with question-level sign-flip permutation p) and the baseline check.
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # efficient_reasoning/
OUT = HERE / "reencode_results"
sys.path.insert(0, str(HERE))
from utils import extract_answer, verify_answer  # noqa: E402


def grade(r):
    pred = extract_answer(r["text"])
    try:
        ok = bool(verify_answer(r["gold"], pred))  # argument order as in their evaluate.py
    except Exception:
        ok = False
    return pred, ok


def sign_p(w, l):
    n = w + l
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n) if n else 1.0


def flip_p(x, n=20000, seed=0):
    rng = random.Random(seed)
    obs = abs(sum(x) / len(x))
    hits = sum(abs(sum(v if rng.random() < .5 else -v for v in x) / len(x)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def main(prefix="math100"):
    recs = {}
    for fn in sorted(OUT.glob(f"{prefix}.*.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            recs[r["uid"]] = r
    (OUT / "eval").mkdir(exist_ok=True)
    G = {}
    with open(OUT / "eval" / f"{prefix}.jsonl", "w") as f:
        for r in recs.values():
            pred, ok = grade(r)
            G[(r["arm"], r["alpha"], r["idx"])] = (ok, r["reasoning_length"], r["hit_eos"])
            f.write(json.dumps({k: r[k] for k in ("uid", "idx", "arm", "alpha", "gold", "reasoning_length", "n_gen",
                                                  "hit_eos")} | {"prediction": pred, "correct": ok}) + "\n")
    cells = defaultdict(list)
    for (arm, a, i), v in G.items():
        cells[(arm, a)].append((i, v))
    print(f"{'arm':8s} {'alpha':>5s} {'n':>4s} {'acc':>6s} {'mean len':>9s} {'caps':>4s}")
    for (arm, a) in sorted(cells, key=lambda k: (["vanilla", "add", "add_re"].index(k[0]), k[1])):
        v = [x for _, x in cells[(arm, a)]]
        print(f"{arm:8s} {a:5.2f} {len(v):4d} {100*sum(x[0] for x in v)/len(v):5.1f}% {sum(x[1] for x in v)/len(v):9.0f} "
              f"{sum(not x[2] for x in v):4d}")
    print("\npaired add_re vs add (same questions):")
    for a in sorted({a for arm, a in cells if arm == "add_re"}):
        Q = sorted(i for (arm, aa, i) in G if arm == "add_re" and aa == a and ("add", a, i) in G)
        if not Q:
            continue
        w = sum(G[("add_re", a, i)][0] and not G[("add", a, i)][0] for i in Q)
        l = sum(G[("add", a, i)][0] and not G[("add_re", a, i)][0] for i in Q)
        d = [G[("add_re", a, i)][1] - G[("add", a, i)][1] for i in Q]
        lr = [math.log(max(G[("add_re", a, i)][1], 1) / max(G[("add", a, i)][1], 1)) for i in Q]
        print(f"  alpha {a:4.2f} n={len(Q):3d} | acc add_re {sum(G[('add_re',a,i)][0] for i in Q)} vs add "
              f"{sum(G[('add',a,i)][0] for i in Q)}, only {w}/{l} sign p {sign_p(w, l):.3f} | mean len diff "
              f"{sum(d)/len(d):+.0f} p {flip_p(d):.3f} | geo ratio {math.exp(sum(lr)/len(lr)):.3f} p {flip_p(lr):.3f}")
    # baseline check: the add baseline must reproduce Steer2Edit's trend (length falls with alpha)
    van = [x for _, x in cells.get(("vanilla", 0.0), [])]
    if van:
        vq = {i: x for i, x in cells[("vanilla", 0.0)]}
        print("\nbaseline check (add vs vanilla, same questions): mean length ratio by alpha")
        for a in sorted(a for arm, a in cells if arm == "add"):
            q = [(i, x) for i, x in cells[("add", a)] if i in vq]
            if q:
                print(f"  alpha {a:4.2f} n={len(q):3d} length ratio {sum(x[1] for _, x in q)/sum(vq[i][1] for i, _ in q):.3f}"
                      f" acc {sum(x[0] for _, x in q)}/{len(q)} vs vanilla {sum(vq[i][0] for i, _ in q)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
