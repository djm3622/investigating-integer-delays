"""Figure 2: how often the loss slope points toward the true delay, by loss and excitation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.figures.signals import magnitude, make_loss, single_source_cases, synth

LOSSES = ("log", "lin", "tilt", "whitened")
STEP = 1e-3
DRAWS = 150  # delays per case and regime


def directions(case: dict) -> list[dict]:
    """Fraction of drawn delays L whose slope points to the truth (K, a and p at truth)."""
    rng = np.random.default_rng(1000 * case["instance"] + (case["kind"] == "triangle"))
    L0, p, K, a = float(case["L"]), float(case["p"]), case["K"], case["a"]
    target = magnitude(synth(case["kind"], L0, p, K, a))
    losses = [make_loss(name, target) for name in LOSSES]

    def toward(L):
        spectra = [magnitude(synth(case["kind"], L + s * STEP, p, K, a)) for s in (1, -1)]
        return [(f(spectra[0]) - f(spectra[1])) * (L - L0) > 0 for f in losses]

    rows = []
    for regime, (low, high) in (("coarse", (100.0, 200.0)),
                                ("fine", (max(100.0, 0.9 * L0), min(200.0, 1.1 * L0)))):
        delays = rng.uniform(low, high, 4 * DRAWS)
        delays = delays[np.abs(delays - L0) >= 0.5][:DRAWS]
        fractions = np.mean([toward(L) for L in delays], axis=0)
        rows += [{**case, "loss": name, "regime": regime, "toward_truth": float(value)}
                 for name, value in zip(LOSSES, fractions)]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/figures/slope_direction.pdf"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for result in pool.map(directions, single_source_cases()) for row in result]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for axis, kind in zip(axes, ("comb", "triangle")):
        for offset, regime in ((-0.2, "coarse"), (0.2, "fine")):
            def mean(name, keep=lambda K: True):
                values = [r["toward_truth"] for r in rows if r["kind"] == kind and r["loss"] == name
                          and r["regime"] == regime and keep(r["K"])]
                return np.mean(values) if values else np.nan
            x = np.arange(len(LOSSES)) + offset
            axis.bar(x, [mean(name) for name in LOSSES], 0.38, label=regime)
            axis.plot(x, [mean(name, lambda K: K < 0.9) for name in LOSSES], "k^", ms=4,
                      label="K < 0.9" if regime == "fine" else None)
            axis.plot(x, [mean(name, lambda K: K >= 0.96) for name in LOSSES], "kv", ms=4, mfc="w",
                      label="K ≥ 0.96" if regime == "fine" else None)
            print(kind, regime, *(f"{name} {mean(name):.2f}" for name in LOSSES))
        axis.axhline(0.5, color="k", ls="--", lw=0.8)
        axis.set_xticks(range(len(LOSSES)), LOSSES)
        axis.set_title(kind)
    axes[0].set(ylabel="Slope toward the true delay", ylim=(0, 1.02))
    axes[0].legend(fontsize=6, loc="upper left")
    figure.tight_layout()
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
