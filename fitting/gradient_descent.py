"""Gradient descent with restarts on Lagrange-relaxed delays.

Each start draws every source's continuous (L, p) uniformly, sets K = 0.9 and a = 0.5, and runs Adam
through fifth-order Lagrange fractional delays (gd_steps steps, one charged synthesis per source per
step, plus one evaluation at the rounded delays). As many starts run as fit in search_syntheses * S
charged syntheses. The start with the lowest loss at rounded delays is kept. A polish then tries
L +/- 1 and the mirrored position of each source, refitting that source's K and a, and the K/a
refinement follows. Every stage uses the same loss.
"""

import copy
import math

import torch
from torch import nn

from fitting.common import Log, refine, set_a, set_K
from model.kps.losses import spectral_loss
from model.kps.pluck import KarplusStrongPluckLagrange
from model.kps.triangle import KarplusStrongTriangleLagrange

MARGIN, K_MAX, A_MAX = 3, 0.9995, 0.99
CHECKPOINTS = (0.05, 0.1, 0.25, 0.5, 1.0)


def _cosine(start: float, end: float, step: int, steps: int) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * step / max(steps - 1, 1)))


def _caps(model):
    """Logit upper bounds for K <= K_MAX and a <= A_MAX; the lower bounds come from the scaling."""
    unit = (K_MAX - model.gain_min) / (model.gain_max - model.gain_min)
    return math.log(unit / (1 - unit)), math.log(A_MAX / (1 - A_MAX))


@torch.no_grad()
def _set_ka(model, K: float, a: float) -> None:
    set_K(model, K)
    set_a(model, a)


@torch.no_grad()
def _clamp_ka(model):
    k_cap, a_cap = _caps(model)
    model.delay_gain.clamp_(max=k_cap)
    model.a.clamp_(max=a_cap)


def _draw_start(settings: dict, sources: int):
    starts = []
    for _ in range(sources):
        L = settings["L_min"] + (settings["L_max"] - settings["L_min"]) * float(torch.rand(()))
        high = min(L - MARGIN, settings["position_max"])
        starts.append((L, MARGIN + (high - MARGIN) * float(torch.rand(()))))
    return starts


def _rounded(model, valid) -> tuple[int, int]:
    L = int(round(float(model.delay_len_parameter)))
    p = min(max(int(round(float(model.position_parameter))), MARGIN), L - MARGIN)
    if (L, p) not in valid:
        raise ValueError(f"Rounded pair {(L, p)} is outside the integer grid.")
    return L, p


def _descend(kind, settings, cfg, loss_of, sources, valid):
    """One start: Adam on (L, p) at 0.3 and (K, a) at 0.01, cosine to 0.01 and 0.001."""
    n = cfg["n_fft"]
    model_type = KarplusStrongPluckLagrange if kind == "comb" else KarplusStrongTriangleLagrange
    models = nn.ModuleList(model_type(**settings, L_init=L, position_init=p)
                           for L, p in _draw_start(settings, sources))
    delays = [q for m in models for q in (m.delay_len_parameter, m.position_parameter)]
    gains = [q for m in models for q in (m.delay_gain, m.a)]
    optimizer = torch.optim.Adam([{"params": delays}, {"params": gains}])
    steps = cfg["gd_steps"]
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = _cosine(0.3, 0.01, step, steps)
        optimizer.param_groups[1]["lr"] = _cosine(0.01, 0.001, step, steps)
        optimizer.zero_grad()
        loss_of(sum(m.lagrange_synth(n) for m in models)).backward()
        optimizer.step()
        with torch.no_grad():
            for m in models:
                m.delay_len_parameter.clamp_(settings["L_min"], settings["L_max"])
                m.position_parameter.clamp_(MARGIN, min(float(m.delay_len_parameter) - MARGIN,
                                                        settings["position_max"]))
                _clamp_ka(m)
    with torch.no_grad():
        pairs = [_rounded(m, valid[i]) for i, m in enumerate(models)]
        for m, (L, p) in zip(models, pairs):
            m.delay_len_parameter.fill_(L)
            m.position_parameter.fill_(p)
        loss = float(loss_of(sum(m.lagrange_synth(n) for m in models)))
    return (loss, pairs, [float(m.scaled_gain().detach()) for m in models],
            [float(m.scaled_allpass().detach()) for m in models])


def _polish(models, pairs, loss_of, n, valid, calls):
    """For each source (two passes when S > 1), try L +/- 1 and the mirrored position."""
    sources = len(models)
    for _ in range(2 if sources > 1 else 1):
        for i in range(sources):
            with torch.no_grad():
                rest = sum((models[j].time_domain_synth(n, *pairs[j]) for j in range(sources) if j != i),
                           torch.zeros(n))
            calls += sources - 1
            L, p = pairs[i]
            candidates = []
            for dL in (0, -1, 1):
                for q in (p, L + dL - p):
                    pair = (L + dL, q)
                    if MARGIN <= q <= L + dL - MARGIN and pair in valid[i] and pair not in candidates:
                        candidates.append(pair)
            K0, best = float(models[i].scaled_gain().detach()), None
            for pair in candidates:
                for a0 in (0.2, 0.7):
                    _set_ka(models[i], K0, a0)
                    optimizer = torch.optim.Adam([{"params": [models[i].delay_gain]},
                                                  {"params": [models[i].a]}])
                    for step in range(120):
                        optimizer.param_groups[0]["lr"] = _cosine(0.02, 0.001, step, 120)
                        optimizer.param_groups[1]["lr"] = _cosine(0.05, 0.002, step, 120)
                        optimizer.zero_grad()
                        loss_of(rest + models[i].time_domain_synth(n, *pair)).backward()
                        optimizer.step()
                        _clamp_ka(models[i])
                    with torch.no_grad():
                        score = float(loss_of(rest + models[i].time_domain_synth(n, *pair)))
                    calls += 121
                    if best is None or score < best[0]:
                        best = (score, pair, float(models[i].scaled_gain()), float(models[i].scaled_allpass()))
            pairs[i] = best[1]
            _set_ka(models[i], best[2], best[3])
    return pairs, calls


def fit_gradient_descent(models: nn.ModuleList, target, cfg, kind: str, settings: dict, log_path=None):
    """Return (history, charged syntheses); leaves `models` (integer) at the polished, refined pairs."""
    n, sources = cfg["n_fft"], len(models)
    spectrum = torch.fft.rfft(target[:n])
    loss_of = lambda x: spectral_loss(x, spectrum, cfg["loss"])
    valid = [set(m.all_pairs()) for m in models]
    log = Log(log_path, cfg["log_syntheses"])
    restarts = max(cfg["search_syntheses"] // (cfg["gd_steps"] + 1), 1)
    checkpoints = {max(round(f * restarts), 1) for f in CHECKPOINTS}
    calls, losses, best, checkpoint_rows, checkpoint_calls = 0, [], None, [], 0
    for start in range(restarts):
        result = _descend(kind, settings, cfg, loss_of, sources, valid)
        calls += sources * (cfg["gd_steps"] + 1)
        losses.append(result[0])
        if best is None or result[0] < best[1][0]:
            best = (start, result)
        log(calls, result[0], result[1], models, force=True)
        if start + 1 in checkpoints:
            # Polish a copy of the best start so far; its cost is logged, not charged.
            trial = copy.deepcopy(models)
            for i in range(sources):
                _set_ka(trial[i], best[1][2][i], best[1][3][i])
            polished, used = _polish(trial, list(best[1][1]), loss_of, n, valid, 0)
            checkpoint_calls += used
            checkpoint_rows.append({"starts": start + 1, "calls": calls, "seconds": log.rows[-1]["seconds"],
                                    "best_start": best[0], "pairs_before_polish": [list(p) for p in best[1][1]],
                                    "pairs_after_polish": [list(p) for p in polished], "polish_syntheses": used})
    descent_calls = calls
    _, (chosen_loss, pairs, K, a) = best
    before = list(pairs)
    for i in range(sources):
        _set_ka(models[i], K[i], a[i])
    pairs, calls = _polish(models, list(pairs), loss_of, n, valid, calls)
    log.rows.append({"calls": calls, "descent_syntheses": descent_calls,
                     "polish_syntheses": calls - descent_calls,
                     "refine_syntheses": cfg["refine_steps"] * sources,
                     "restarts": restarts, "chosen_restart": best[0],
                     "checkpoints": checkpoint_rows, "checkpoint_polish_syntheses_uncharged": checkpoint_calls,
                     "chosen_restart_loss": chosen_loss, "restart_losses": losses,
                     "pairs_before_polish": [list(p) for p in before],
                     "pairs_after_polish": [list(p) for p in pairs]})
    return log.rows, refine(models, pairs, spectrum, cfg, log, calls)
