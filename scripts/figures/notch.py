"""Figure 1: the log loss along integer delay errors, against the closed form, and its notch width."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.figures.signals import (dilogarithm, infinite_magnitude, magnitude, make_loss, plateau,
                                     single_source_cases, synth)

ERRORS = np.round(np.arange(-8, 8.0001, 0.05), 2)
WINDOWS = {"rectangular": np.ones, "Hann": lambda n: 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n),
           "Blackman": lambda n: (0.42 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
                                  + 0.08 * np.cos(4 * np.pi * np.arange(n) / n))}
LENGTHS = {"rectangular": (2048, 4096, 8192, 16384, 32768), "Hann": (2048, 8192, 32768), "Blackman": (8192,)}


def half_width(loss: np.ndarray) -> float:
    """Half the width of the notch at half the depth of the plateau."""
    half = np.median(loss[np.abs(ERRORS) >= 4]) / 2
    center = int(np.argmin(np.abs(ERRORS)))
    right = left = center
    while right < len(ERRORS) - 1 and loss[right] < half:
        right += 1
    while left > 0 and loss[left] < half:
        left -= 1
    return (ERRORS[right] - ERRORS[left]) / 2


def notch_widths(case: dict) -> dict:
    """Half-width of the log-loss notch per (window, length), and for the infinite response."""
    parameters = case["p"], case["K"], case["a"]
    widths = {}
    for name, lengths in LENGTHS.items():
        for n in lengths:
            window = WINDOWS[name](n)
            loss = make_loss("log", magnitude(synth(case["kind"], case["L"], *parameters, n) * window))
            widths[name, n] = half_width(np.array(
                [loss(magnitude(synth(case["kind"], case["L"] + d, *parameters, n) * window)) for d in ERRORS]))
    loss = make_loss("log", infinite_magnitude(case["kind"], case["L"], *parameters))
    widths["none", 8192] = half_width(np.array(
        [loss(infinite_magnitude(case["kind"], case["L"] + d, *parameters)) for d in ERRORS]))
    return widths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/figures/notch.pdf"))
    parser.add_argument("--comb-instances", type=int, nargs="+", default=[1, 29],
                        help="Single-source comb cases to draw: one well damped, one lightly damped")
    args = parser.parse_args()
    cases = single_source_cases()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for color, instance in zip(("C0", "C3"), args.comb_instances):
        case = next(c for c in cases if c["kind"] == "comb" and c["instance"] == instance)
        K, errors = case["K"], np.arange(-12, 13)
        delays = case["L"] + errors
        errors, delays = errors[(delays >= 100) & (delays <= 200)], delays[(delays >= 100) & (delays <= 200)]
        for spectrum, marker, label in (
                (lambda L: magnitude(synth("comb", L, case["p"], K, case["a"])), "o-", "8,192 samples"),
                (lambda L: infinite_magnitude("comb", L, case["p"], K, case["a"]), "x", "infinite response")):
            loss = make_loss("log_squared", spectrum(case["L"]))
            left.plot(errors, [loss(spectrum(L)) / plateau(K) for L in delays], marker, color=color, ms=4,
                      lw=1, label=f"K = {K:g}, {label}")
        left.plot([-1, 1], [1 - dilogarithm(K * K / 4) / plateau(K)] * 2, "_", color=color, ms=14)
    left.axhline(1, color="k", lw=0.6)
    left.set(xlabel="Integer delay error", ylabel="Squared log loss / P(K)")
    left.legend(fontsize=6)
    widths = [(case["kind"], notch_widths(case)) for case in cases]
    for color, kind in (("C0", "comb"), ("C1", "triangle")):
        for name, style in (("rectangular", "o-"), ("Hann", "s--"), ("Blackman", "^"), ("none", "x")):
            lengths = LENGTHS.get(name, (8192,))
            right.plot(lengths, [np.median([w[name, n] for k, w in widths if k == kind]) for n in lengths], style,
                       color=color, ms=4, lw=1, label=f"{kind}, {'no window' if name == 'none' else name}")
    right.set_xscale("log", base=2)
    right.set(xlabel="Signal length", ylabel="Notch half-width (samples)", ylim=(0, 2))
    right.legend(fontsize=6)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
