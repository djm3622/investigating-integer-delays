"""The three stages of a fit: descent on relaxed delays, integer polish, and K/a refinement.

Every stage takes `loss_of`, the fit's loss of a synthesized window, and `n`, the window length.
"""

import math
from typing import Callable, NamedTuple

import torch
from torch import nn

from model.kps import LAGRANGE_MODELS

Pair = tuple[int, int]
Loss = Callable[[torch.Tensor], torch.Tensor]

MARGIN = 3  # samples kept between p and both ends of the loop
K_MAX, A_MAX = 0.9995, 0.99
POLISH_STEPS = 120
POLISH_A_STARTS = (0.2, 0.7)


class Start(NamedTuple):
    """Outcome of one descent: the loss at the rounded delays, the pairs, and every source's K and a."""
    loss: float
    pairs: list[Pair]
    K: list[float]
    a: list[float]


def gains_of(models: nn.ModuleList) -> tuple[list[float], list[float]]:
    """Every source's K, and every source's a."""
    K, a = zip(*(m.gains() for m in models))
    return list(K), list(a)


def _cosine(start: float, end: float, step: int, steps: int) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * step / max(steps - 1, 1)))


def _draw_pair(settings: dict) -> tuple[float, float]:
    L = settings["L_min"] + (settings["L_max"] - settings["L_min"]) * float(torch.rand(()))
    high = min(L - MARGIN, settings["position_max"])
    return L, MARGIN + (high - MARGIN) * float(torch.rand(()))


def _rounded_pair(model) -> Pair:
    L = int(round(float(model.delay_len_parameter)))
    p = min(max(int(round(float(model.position_parameter))), MARGIN), L - MARGIN)
    if not model.is_valid((L, p)):
        raise ValueError(f"Rounded pair {(L, p)} is outside the integer grid.")
    return L, p


@torch.no_grad()
def _clamp_to_grid(model, settings: dict) -> None:
    model.delay_len_parameter.clamp_(settings["L_min"], settings["L_max"])
    model.position_parameter.clamp_(MARGIN, min(float(model.delay_len_parameter) - MARGIN,
                                                settings["position_max"]))
    model.clamp_gains(K_MAX, A_MAX)


def descend(kind: str, sources: int, settings: dict, loss_of: Loss, n: int, steps: int) -> Start:
    """One start: Adam on (L, p) at 0.3 and (K, a) at 0.01, cosine to 0.01 and 0.001."""
    starts = [_draw_pair(settings) for _ in range(sources)]
    models = nn.ModuleList(LAGRANGE_MODELS[kind](**settings, L_init=L, position_init=p) for L, p in starts)
    delays = [q for m in models for q in (m.delay_len_parameter, m.position_parameter)]
    gains = [q for m in models for q in (m.delay_gain, m.a)]
    optimizer = torch.optim.Adam([{"params": delays}, {"params": gains}])
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = _cosine(0.3, 0.01, step, steps)
        optimizer.param_groups[1]["lr"] = _cosine(0.01, 0.001, step, steps)
        optimizer.zero_grad()
        loss_of(sum(m.lagrange_synth(n) for m in models)).backward()
        optimizer.step()
        for m in models:
            _clamp_to_grid(m, settings)
    with torch.no_grad():
        pairs = [_rounded_pair(m) for m in models]
        for m, (L, p) in zip(models, pairs):
            m.delay_len_parameter.fill_(L)
            m.position_parameter.fill_(p)
        loss = float(loss_of(sum(m.lagrange_synth(n) for m in models)))
    return Start(loss, pairs, *gains_of(models))


def _neighbours(model, pair: Pair) -> list[Pair]:
    """The pair, L +/- 1, and the mirrored position of each, inside the grid."""
    L, p = pair
    candidates = []
    for dL in (0, -1, 1):
        for q in (p, L + dL - p):
            candidate = (L + dL, q)
            if MARGIN <= q <= L + dL - MARGIN and model.is_valid(candidate) and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _fit_gains(model, pair: Pair, K: float, a: float, rest: torch.Tensor, loss_of: Loss,
               n: int) -> tuple[float, Pair, float, float]:
    """Refit one source's K and a at a fixed pair, from (K, a); returns (loss, pair, K, a)."""
    model.set_gains(K, a)
    optimizer = torch.optim.Adam([{"params": [model.delay_gain]}, {"params": [model.a]}])
    for step in range(POLISH_STEPS):
        optimizer.param_groups[0]["lr"] = _cosine(0.02, 0.001, step, POLISH_STEPS)
        optimizer.param_groups[1]["lr"] = _cosine(0.05, 0.002, step, POLISH_STEPS)
        optimizer.zero_grad()
        loss_of(rest + model.time_domain_synth(n, *pair)).backward()
        optimizer.step()
        model.clamp_gains(K_MAX, A_MAX)
    with torch.no_grad():
        loss = float(loss_of(rest + model.time_domain_synth(n, *pair)))
    return loss, pair, *model.gains()


def polish(models: nn.ModuleList, pairs: list[Pair], loss_of: Loss, n: int) -> tuple[list[Pair], int]:
    """For each source (two passes when S > 1), try L +/- 1 and the mirrored position.

    Returns the polished pairs and the syntheses used; leaves each model at its best K and a.
    """
    pairs, sources, syntheses = list(pairs), len(models), 0
    for _ in range(2 if sources > 1 else 1):
        for i, model in enumerate(models):
            with torch.no_grad():
                rest = sum((models[j].time_domain_synth(n, *pairs[j]) for j in range(sources) if j != i),
                           torch.zeros(n))
            K_start, _ = model.gains()
            trials = [_fit_gains(model, pair, K_start, a_start, rest, loss_of, n)
                      for pair in _neighbours(model, pairs[i]) for a_start in POLISH_A_STARTS]
            syntheses += sources - 1 + (POLISH_STEPS + 1) * len(trials)
            _, pairs[i], K, a = min(trials, key=lambda trial: trial[0])
            model.set_gains(K, a)
    return pairs, syntheses


def _reject_nonfinite_gradients(models: nn.ModuleList, step: int) -> None:
    """Reject invalid derivatives before they corrupt optimizer/model state."""
    for name, parameter in models.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Nonfinite gradient for {name} at update {step + 1}; "
                                     "optimizer update rejected.")


def refine(models: nn.ModuleList, pairs: list[Pair], loss_of: Loss, n: int, steps: int, lr: float) -> float:
    """Freeze the pairs and refine every K and a with Adam; returns the loss before the last update."""
    optimizer = torch.optim.Adam([q for m in models for q in (m.delay_gain, m.a)], lr=lr)
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_of(torch.stack([m.time_domain_synth(n, *pair) for m, pair in zip(models, pairs)]).sum(0))
        loss.backward()
        _reject_nonfinite_gradients(models, step)
        optimizer.step()
    return float(loss.detach())
