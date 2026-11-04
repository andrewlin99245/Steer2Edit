"""Grade the AIME runs (edit, edit_re, vanilla) with their extract_answer / verify_answer, unmodified.

    cd efficient_reasoning && python reencode/evaluate_edit_aime.py      (needs math_verify only; no GPU)
Writes reencode_results/edit/eval/aime.jsonl; prints per setting: correct, average length, cap hits, then edit_re vs edit
pairs and each setting vs vanilla, on the questions both have finished.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_reencode import OUT as S2E_OUT, flip_p, grade, sign_p  # noqa: E402

OUT = S2E_OUT / "edit"
PAIRS = [("r0.65", "r0.70"), ("r0.80", "r0.80"), ("r0.80", "r1.00"), ("r1.00", "r1.00")]


def main():
    recs = {}
    for fn in sorted(OUT.glob("aime.*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            recs[r["uid"]] = r
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    G = {}
    with open(OUT / "eval" / "aime.jsonl", "w") as f:
        for r in recs.values():
            pred, ok = grade(r)
            G[(r["arm"], r["cfg"].split("_")[0], r["idx"])] = (ok, r["reasoning_length"], r["hit_eos"])
            f.write(json.dumps({k: r[k] for k in ("uid", "idx", "arm", "cfg", "gold", "reasoning_length", "n_gen", "hit_eos",
                                                  "n_restores")} | {"prediction": pred, "correct": ok}) + "\n")
    print(f"{len(recs)} runs")
    for arm, rho in sorted({k[:2] for k in G}, key=lambda x: (x[1], x[0])):
        v = [x for k, x in G.items() if k[:2] == (arm, rho)]
        print(f"{arm:7s} {rho} n={len(v):3d} correct {sum(x[0] for x in v):3d} avg len {sum(x[1] for x in v) / len(v):6.0f} "
              f"caps {sum(not x[2] for x in v)}")
    for a, b in PAIRS:
        Q = sorted(k[2] for k in G if k[:2] == ("edit_re", a) and ("edit", b, k[2]) in G)
        if Q:
            A, B = [G[("edit_re", a, i)] for i in Q], [G[("edit", b, i)] for i in Q]
            w, l = sum(x[0] and not y[0] for x, y in zip(A, B)), sum(y[0] and not x[0] for x, y in zip(A, B))
            d = [x[1] - y[1] for x, y in zip(A, B)]
            print(f"edit_re {a} vs edit {b} n={len(Q)}: correct {sum(x[0] for x in A)} vs {sum(y[0] for y in B)}, only {w}/{l} "
                  f"sign p {sign_p(w, l):.3f} | avg len diff {sum(d) / len(d):+.0f} p {flip_p(d):.3f}")
    for b in ("edit", "edit_re"):  # edit_ent at rho 1.00
        Q = sorted(k[2] for k in G if k[:2] == ("edit_ent", "r1.00") and (b, "r1.00", k[2]) in G)
        if Q:
            A, B = [G[("edit_ent", "r1.00", i)] for i in Q], [G[(b, "r1.00", i)] for i in Q]
            w, l = sum(x[0] and not y[0] for x, y in zip(A, B)), sum(y[0] and not x[0] for x, y in zip(A, B))
            d = [x[1] - y[1] for x, y in zip(A, B)]
            print(f"edit_ent r1.00 vs {b} r1.00 n={len(Q)}: correct {sum(x[0] for x in A)} vs {sum(y[0] for y in B)}, only {w}/{l} "
                  f"sign p {sign_p(w, l):.3f} | avg len diff {sum(d) / len(d):+.0f} p {flip_p(d):.3f}")
    V = {k[2]: x for k, x in G.items() if k[:2] == ("vanilla", "vanilla")}
    for arm, rho in sorted({k[:2] for k in G if k[0] != "vanilla"}, key=lambda x: (x[1], x[0])):
        Q = sorted(k[2] for k in G if k[:2] == (arm, rho) and k[2] in V)
        if Q:
            A, B = [G[(arm, rho, i)] for i in Q], [V[i] for i in Q]
            w, l = sum(x[0] and not y[0] for x, y in zip(A, B)), sum(y[0] and not x[0] for x, y in zip(A, B))
            d = [x[1] - y[1] for x, y in zip(A, B)]
            print(f"{arm} {rho} vs vanilla n={len(Q)}: correct {sum(x[0] for x in A)} vs {sum(y[0] for y in B)}, only {w}/{l} "
                  f"sign p {sign_p(w, l):.3f} | avg len diff {sum(d) / len(d):+.0f} p {flip_p(d):.3f}")


if __name__ == "__main__":
    main()
