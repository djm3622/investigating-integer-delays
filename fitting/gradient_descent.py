"""Gradient descent with restarts on Lagrange-relaxed delays.

Each start draws every source's continuous (L, p) uniformly, sets K = 0.9 and a = 0.5, and runs Adam
through fifth-order Lagrange fractional delays (gd_steps steps, one charged synthesis per source per
step, plus one evaluation at the rounded delays). As many starts run as fit in search_syntheses * S
charged syntheses. The start with the lowest loss at rounded delays is kept. A polish then tries
L +/- 1 and the mirrored position of each source, refitting that source's K and a, and the K/a
refinement follows. Every stage uses the same loss.
"""

import copy
import json
from pathlib import Path
import time
from typing import NamedTuple

import torch
from torch import nn

from fitting.losses import spectral_loss
from fitting.stages import Loss, Pair, Start, descend, gains_of, polish, refine
from model.kps import INTEGER_MODELS

CHECKPOINTS = (0.05, 0.1, 0.25, 0.5, 1.0)  # fractions of the restarts at which the best start is polished


class Fit(NamedTuple):
    """Estimates of every source, the history rows, and the charged syntheses."""
    pairs: list[Pair]
    K: list[float]
    a: list[float]
    history: list[dict]
    syntheses: int


class History:
    """Rows of a fit; progress rows are also appended to a JSONL file when a path is given."""

    def __init__(self, path: Path | None) -> None:
        self.path, self.rows = path, []
        self.start = time.perf_counter()  # rows carry seconds since the fit started

    def progress(self, calls: int, loss: float, pairs: list[Pair], models: nn.ModuleList) -> None:
        K, a = gains_of(models)
        row = {"calls": calls, "seconds": time.perf_counter() - self.start, "loss": float(loss),
               "pairs": _lists(pairs), "K": K, "a": a}
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")


def _lists(pairs: list[Pair]) -> list[list[int]]:
    return [list(pair) for pair in pairs]


def _polish_start(models: nn.ModuleList, start: Start, loss_of: Loss, n: int) -> tuple[list[Pair], int]:
    """Polish a start's pairs from its K and a."""
    for model, K, a in zip(models, start.K, start.a):
        model.set_gains(K, a)
    return polish(models, start.pairs, loss_of, n)


def fit_gradient_descent(kind: str, sources: int, target: torch.Tensor, loss: str, model_settings: dict,
                         fit_settings: dict, log_path: Path | None = None) -> Fit:
    """Fit `sources` strings of excitation `kind` to the first n_fft samples of the target."""
    n, steps = fit_settings["n_fft"], fit_settings["gd_steps"]
    spectrum = torch.fft.rfft(target[:n])
    loss_of = lambda x: spectral_loss(x, spectrum, loss)
    models = nn.ModuleList(INTEGER_MODELS[kind](**model_settings) for _ in range(sources))
    history = History(log_path)
    restarts = max(fit_settings["search_syntheses"] // (steps + 1), 1)
    checkpoints = {max(round(fraction * restarts), 1) for fraction in CHECKPOINTS}
    calls, starts, best, checkpoint_rows = 0, [], 0, []
    for index in range(restarts):
        starts.append(descend(kind, sources, model_settings, loss_of, n, steps))
        calls += sources * (steps + 1)
        if starts[index].loss < starts[best].loss:
            best = index
        # K and a of these rows are the models' initial values: the models are first set by the polish.
        history.progress(calls, starts[index].loss, starts[index].pairs, models)
        if index + 1 in checkpoints:
            # Polish a copy of the best start so far; its cost is logged, not charged.
            polished, used = _polish_start(copy.deepcopy(models), starts[best], loss_of, n)
            checkpoint_rows.append({"starts": index + 1, "calls": calls, "seconds": history.rows[-1]["seconds"],
                                    "best_start": best, "pairs_before_polish": _lists(starts[best].pairs),
                                    "pairs_after_polish": _lists(polished), "polish_syntheses": used})
    pairs, polish_calls = _polish_start(models, starts[best], loss_of, n)
    refine_calls = fit_settings["refine_steps"] * sources
    history.rows.append({"calls": calls + polish_calls, "descent_syntheses": calls,
                         "polish_syntheses": polish_calls, "refine_syntheses": refine_calls,
                         "restarts": restarts, "chosen_restart": best, "checkpoints": checkpoint_rows,
                         "checkpoint_polish_syntheses_uncharged": sum(row["polish_syntheses"]
                                                                      for row in checkpoint_rows),
                         "chosen_restart_loss": starts[best].loss,
                         "restart_losses": [start.loss for start in starts],
                         "pairs_before_polish": _lists(starts[best].pairs),
                         "pairs_after_polish": _lists(pairs)})
    calls += polish_calls + refine_calls
    final_loss = refine(models, pairs, loss_of, n, fit_settings["refine_steps"], fit_settings["refine_lr"])
    history.progress(calls, final_loss, pairs, models)
    return Fit(pairs, *gains_of(models), history.rows, calls)
