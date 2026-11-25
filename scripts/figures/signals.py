"""Double-precision signals, losses and closed form shared by the notch and slope figures."""

import re
from math import comb

import numpy as np
from scipy.signal import lfilter
import yaml

from data.cases import ROOT, make_cases
from fitting.losses import EPS, TILT_CUTOFF

N = 8192


def single_source_cases() -> list[dict]:
    """All 32 single-source cases per excitation, with their parameters from the file names."""
    config = yaml.safe_load((ROOT / "configs/loss_comparison.yaml").read_text())
    config |= {"source_counts": [1], "instances": config["instances_drawn"]}
    cases = []
    for case in make_cases(config):
        K, a, L, p = re.search(r"gain_([\d.]+)_a_([\d.]+)_L_(\d+)_pos_[\d.]+_[dA]_(\d+)",
                               case["paths"][0].stem).groups()
        cases.append({"kind": case["filter"], "instance": case["instance"],
                      "K": float(K), "a": float(a), "L": int(L), "p": int(p)})
    return cases


def lagrange(delay: float, order: int = 5) -> np.ndarray:
    nodes = np.arange(order + 1)
    return np.array([np.prod((delay - nodes[nodes != k]) / (k - nodes[nodes != k]))
                     for k in range(order + 1)])


def excitation(kind: str, L: float, p: float, n: int) -> np.ndarray:
    if kind == "comb":
        e = np.zeros(n)
        e[0] = 1.0
        start = int(np.floor(p)) - 2
        e[start:start + 6] -= lagrange(p - start)
        return e
    t = np.arange(n, dtype=float)
    return np.where(t <= p, t / p, np.clip((L - t) / (L - p), 0, None))


def synth(kind: str, L: float, p: float, K: float, a: float, n: int = N) -> np.ndarray:
    """Causal response, truncated to n samples; fractional delays through fifth-order Lagrange."""
    numerator = np.array([a / 2, (a + 1) / 2, 0.5])
    start = int(np.floor(L)) - 2
    fir = np.convolve(numerator, lagrange(L - start))
    denominator = np.zeros(start + len(fir))
    denominator[0] = 1
    denominator[1] += a
    denominator[start:] -= K * fir
    return lfilter(numerator, denominator, -K * excitation(kind, L, p, n))


def infinite_magnitude(kind: str, L: float, p: float, K: float, a: float, n: int = N) -> np.ndarray:
    """|X| of the infinite response on the n-point rFFT bins: no truncation, no leakage."""
    w = 2 * np.pi * np.arange(n // 2 + 1) / n
    z = np.exp(-1j * w)
    numerator = a / 2 + (a + 1) / 2 * z + 0.5 * z ** 2
    denominator = 1 + a * z - K * numerator * np.exp(-1j * w * L)
    E = 1 - np.exp(-1j * w * p) if kind == "comb" else np.fft.rfft(excitation(kind, L, p, n))
    return np.abs(-K * E * numerator / denominator)


def magnitude(x: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(x))


def make_loss(name: str, Y: np.ndarray):
    """Loss of a magnitude spectrum against the target magnitude Y."""
    if name == "log_squared":
        return lambda M: np.mean((np.log(M + EPS) - np.log(Y + EPS)) ** 2)
    if name == "log":
        return lambda M: np.mean(np.abs(10 * np.log10(M + EPS) - 10 * np.log10(Y + EPS)))
    if name == "lin":
        return lambda M: np.mean(np.abs(M - Y)) / Y.mean()
    beta = {"tilt": 2, "whitened": -2}[name]  # magnitudes weighted by about omega^-beta
    omega = np.arange(len(Y)) * np.pi / (len(Y) - 1)
    weight = (omega ** 2 + TILT_CUTOFF ** 2) ** (-beta / 2)
    weighted = Y * weight
    return lambda M: np.mean(np.abs(M * weight - weighted)) / weighted.mean()


def plateau(K: float, terms: int = 400) -> float:
    """P(K), the squared log loss two or more samples from the true delay."""
    return sum((K * K / 4) ** m * comb(2 * m, m) / m ** 2 for m in range(1, terms))


def dilogarithm(x: float, terms: int = 400) -> float:
    return sum(x ** m / m ** 2 for m in range(1, terms))
