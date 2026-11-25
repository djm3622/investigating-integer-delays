"""Score the loss comparison: set match per cell, paired tests, and the restart law."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from data.cases import ROOT

rng = np.random.default_rng(20261119)
GRID = np.linspace(0.001, 0.999, 999)


def set_match(true, found, kind: str, canonical: bool = False) -> float:
    """Fraction of true (L, p) pairs found, as multisets; canonical counts p and L - p as one (triangle)."""
    def key(pair):
        L, p = int(pair[0]), int(pair[1])
        return (L, min(p, L - p)) if canonical and kind == "triangle" else (L, p)
    t, f = Counter(map(key, true)), Counter(map(key, found))
    return sum((t & f).values()) / len(true)


def restart_rate(hits: dict[int, tuple[int, int]]) -> float:
    """Per-source rate q of Q_S = q^S by maximum likelihood; hits maps S to (all-right starts, starts)."""
    likelihood = np.zeros_like(GRID)
    for S, (right, starts) in hits.items():
        Q = np.clip(GRID ** S, 1e-12, 1 - 1e-12)
        likelihood += right * np.log(Q) + (starts - right) * np.log(1 - Q)
    return float(GRID[likelihood.argmax()])


def holm(p_values: list[float]) -> list[float]:
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, k in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[k]))
        adjusted[k] = running
    return adjusted


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    """95% interval of the mean, resampling the values."""
    means = rng.choice(values, (10000, len(values))).mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def load(root: Path) -> list[dict]:
    """One row per finished fit, with how many of its starts had every source right before the polish."""
    fits = []
    for path in sorted(root.glob("*/*/n*/i*/gd/summary.json")):
        s = json.loads(path.read_text())
        true = [(t["L"], t["position"]) for t in s["true"]]
        starts = s["history"][:next(h["restarts"] for h in s["history"] if "restarts" in h)]
        fits.append({"loss": s["loss"], "filter": s["filter"], "S": s["source_count"], "snr_db": s["snr_db"],
                     "instance": s["instance"], "seconds": s["fit_seconds"],
                     "set_match": set_match(true, s["found"], s["filter"]),
                     "canonical_set_match": set_match(true, s["found"], s["filter"], True),
                     "right": sum(set_match(true, h["pairs"], s["filter"]) == 1.0 for h in starts),
                     "starts": len(starts)})
    return fits


def cells_of(fits: list[dict], kind: str) -> dict[str, dict[tuple, dict]]:
    """The fits of one excitation: loss -> (S, noise, instance) -> fit."""
    cells = defaultdict(dict)
    for f in fits:
        if f["filter"] == kind:
            cells[f["loss"]][f["S"], f["snr_db"], f["instance"]] = f
    return cells


def print_set_match(cells: dict) -> None:
    for loss, rows in cells.items():
        by_S = defaultdict(list)
        for (S, _, _), f in rows.items():
            by_S[S].append(f["set_match"])
        print(f"  {loss:8s} n {len(rows):3d}  {np.mean([f['set_match'] for f in rows.values()]):.3f} "
              f"[{np.mean([f['canonical_set_match'] for f in rows.values()]):.3f}]  "
              + "  ".join(f"S{S} {np.mean(v):.2f}" for S, v in sorted(by_S.items())))


def print_paired_tests(cells: dict, reference: str) -> None:
    """Set match of each loss minus the reference's over shared cells: bootstrap interval, Wilcoxon, Holm."""
    tests = []
    for loss, rows in cells.items():
        shared = sorted(set(rows) & set(cells.get(reference, {})), key=str)
        if loss == reference or len(shared) < 5:
            continue
        d = np.array([rows[k]["set_match"] - cells[reference][k]["set_match"] for k in shared])
        tests.append((loss, d.mean(), *bootstrap_interval(d),
                      float(wilcoxon(d, zero_method="zsplit").pvalue) if d.any() else 1.0))
    for (loss, mean, low, high, p), adjusted in zip(tests, holm([t[-1] for t in tests])):
        print(f"    {loss} - {reference}: {mean:+.3f} [{low:+.3f}, {high:+.3f}]  p {p:.2g}  Holm {adjusted:.2g}")


def restart_table(fits: list[dict]) -> list[dict]:
    """Q_S per loss, excitation and S, with bootstrap intervals over fits, and the fitted q."""
    groups = defaultdict(lambda: defaultdict(list))
    for f in fits:
        groups[f["loss"], f["filter"]][f["S"]].append((f["right"], f["starts"]))
    totals = lambda rows: (sum(r for r, _ in rows), sum(n for _, n in rows))
    table = []
    for (loss, kind), by_S in groups.items():
        q = restart_rate({S: totals(rows) for S, rows in by_S.items()})
        draws = [restart_rate({S: totals([rows[j] for j in rng.integers(0, len(rows), len(rows))])
                               for S, rows in by_S.items()}) for _ in range(2000)]
        for S, rows in sorted(by_S.items()):
            rates = np.array([r / n for r, n in rows])
            low, high = bootstrap_interval(rates)
            table.append({"loss": loss, "filter": kind, "S": S, "fits": len(rows), "starts": totals(rows)[1],
                          "Q_S": rates.mean(), "Q_low": low, "Q_high": high, "q": q,
                          "q_low": np.percentile(draws, 2.5), "q_high": np.percentile(draws, 97.5)})
    return table


def print_restart_law(table: list[dict]) -> None:
    print("\nRestart law, Q_S = q^S (all S sources right in one start, before polish)")
    for loss, kind in dict.fromkeys((r["loss"], r["filter"]) for r in table):
        rows = [r for r in table if (r["loss"], r["filter"]) == (loss, kind)]
        print(f"  {kind:8s} {loss:8s} q {rows[0]['q']:.3f} [{rows[0]['q_low']:.3f}, {rows[0]['q_high']:.3f}]  "
              + "  ".join(f"S{r['S']} {r['Q_S']:.3f}" for r in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/loss_comparison")
    parser.add_argument("--reference", default="lin", help="Loss the paired tests compare against")
    args = parser.parse_args()
    fits = load(args.input)
    if not fits:
        parser.error(f"no finished fits under {args.input}")
    write_csv(args.input / "fits.csv", fits)
    for kind in sorted({f["filter"] for f in fits}):
        cells = cells_of(fits, kind)
        print(f"\n{kind}: set match pooled over S and noise [canonical], then by S")
        print_set_match(cells)
        print_paired_tests(cells, args.reference)
    table = restart_table(fits)
    write_csv(args.input / "restart_law.csv", table)
    print_restart_law(table)


if __name__ == "__main__":
    main()
