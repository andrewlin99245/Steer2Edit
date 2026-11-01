"""Accuracy-vs-length curves of edit and edit_re at alpha 0.05.

    python reencode/evaluate_edit_curve.py     (after evaluate_edit_reencode.py; standard library only)
Reads the graded reencode_results/edit/eval/math100.jsonl. For each arm: piecewise-linear accuracy as a function of mean
reasoning length through its settings; gap = edit_re - edit, averaged over the overlapping length range; 95% bootstrap CI
over questions (the same resampled questions for every setting and arm, so pairing is kept).
Exploratory: settings were not chosen on held-out questions and nothing is corrected for multiple comparisons.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "reencode_results" / "edit" / "eval" / "math100.jsonl"
ARMS = ("edit", "edit_re")
N_GRID, N_BOOT, SEED = 50, 2000, 64


def curve(table, arm, qs):
    """Sorted (mean length, accuracy, cfg) points of one arm on the question list qs (repeats allowed)."""
    pts = []
    for cfg, rows in table[arm].items():
        pts.append((sum(rows[q][1] for q in qs) / len(qs), 100 * sum(rows[q][0] for q in qs) / len(qs), cfg))
    return sorted(pts)


def at(pts, x):
    for (x0, y0, _), (x1, y1, _) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise ValueError(x)


def gap(table, qs):
    a, b = curve(table, "edit_re", qs), curve(table, "edit", qs)
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if lo >= hi:
        return None, lo, hi
    xs = [lo + (hi - lo) * i / (N_GRID - 1) for i in range(N_GRID)]
    return sum(at(a, x) - at(b, x) for x in xs) / N_GRID, lo, hi


def main():
    table = {a: defaultdict(dict) for a in ARMS}
    for line in open(EVAL):
        r = json.loads(line)
        if r["arm"] in ARMS and r["cfg"].endswith("_a0.05"):
            table[r["arm"]][r["cfg"]][r["idx"]] = (r["correct"], r["reasoning_length"])
    rows = [v for a in ARMS for v in table[a].values()]
    qs = sorted(set.intersection(*(set(v) for v in rows)))
    print(f"questions in every setting: {len(qs)}; missing runs per setting: "
          f"{ {f'{a} {c}': len(qs_all) - len(qs) for a in ARMS for c, qs_all in table[a].items() if len(qs_all) != len(qs)} }")
    for a in ARMS:
        print(f"{a}: " + "  ".join(f"{c.split('_')[0]} {y:.0f}%/{x:.0f}" for x, y, c in curve(table, a, qs)))
    g, lo, hi = gap(table, qs)
    if g is None:
        print("no overlapping length range")
        return
    rng, boots, skipped = random.Random(SEED), [], 0
    for _ in range(N_BOOT):
        v = gap(table, rng.choices(qs, k=len(qs)))[0]
        if v is None:
            skipped += 1
        else:
            boots.append(v)
    boots.sort()
    print(f"gap edit_re - edit over lengths {lo:.0f}-{hi:.0f}: {g:+.2f} points, 95% CI "
          f"[{boots[int(.025 * len(boots))]:+.2f}, {boots[int(.975 * len(boots))]:+.2f}] "
          f"({N_BOOT} resamples, {skipped} without overlap)")


if __name__ == "__main__":
    main()
