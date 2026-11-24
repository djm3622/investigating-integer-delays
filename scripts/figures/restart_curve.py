"""Figure 3: share of starts with every source right against the number of sources."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.cases import ROOT

CURVES = (("comb", "tilt", "C0"), ("triangle", "lin", "C1"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/loss_comparison/restart_law.csv")
    parser.add_argument("--output", type=Path, default=Path("output/figures/restart_curve.pdf"))
    args = parser.parse_args()
    with args.input.open() as stream:
        table = list(csv.DictReader(stream))
    figure, axis = plt.subplots(figsize=(3.5, 2.6))
    for kind, loss, color in CURVES:
        rows = [r for r in table if (r["filter"], r["loss"]) == (kind, loss)]
        S, Q, low, high = (np.array([float(r[key]) for r in rows]) for key in ("S", "Q_S", "Q_low", "Q_high"))
        q = float(rows[0]["q"])
        axis.errorbar(S, Q, [Q - low, high - Q], fmt="o-", color=color, ms=4, lw=1, capsize=2,
                      label=f"{kind}, {loss}")
        axis.plot(S, q ** S, "--", color=color, lw=1, label=f"q = {q:.3f}")
    axis.set_yscale("log")
    axis.set(xlabel="Number of sources S", ylabel="$Q_S$")
    axis.secondary_yaxis("right", functions=(lambda Q: np.log(20) / np.maximum(Q, 1e-9),
                                             lambda R: np.log(20) / np.maximum(R, 1e-9))
                         ).set_ylabel("Restarts for 95%")
    axis.legend(fontsize=6)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
