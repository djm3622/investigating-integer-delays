"""Pieces shared by the two solvers."""

import json
from pathlib import Path
import time

import torch
from torch import nn

from model.kps.losses import spectral_loss


def set_K(model, K: float) -> None:
    unit = min(max((K - model.gain_min) / (model.gain_max - model.gain_min), 1e-4), 1 - 1e-4)
    model.delay_gain.copy_(torch.logit(model.delay_gain.new_tensor(unit)))


def set_a(model, a: float) -> None:
    model.a.copy_(torch.logit(model.a.new_tensor(min(max(a, 1e-4), 1 - 1e-4))))


class Log:
    """Best-so-far rows, one every ``every`` charged syntheses, optionally appended to a JSONL file."""

    def __init__(self, path: Path | None, every: int):
        self.path, self.every, self.rows, self.next = path, every, [], 0
        self.start = time.perf_counter()  # rows carry seconds since the fit started

    def __call__(self, calls, loss, pairs, models, force=False):
        if calls < self.next and not force:
            return
        self.next = calls + self.every
        row = {"calls": calls, "seconds": time.perf_counter() - self.start, "loss": float(loss),
               "pairs": [list(p) for p in pairs], "K": [float(m.scaled_gain().detach()) for m in models],
               "a": [float(m.scaled_allpass().detach()) for m in models]}
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")


def checked_step(optimizer: torch.optim.Optimizer, models: nn.ModuleList, step: int) -> None:
    """Reject invalid derivatives before they corrupt optimizer/model state."""
    for name, parameter in models.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Nonfinite gradient for {name} at update {step + 1}; "
                                     "optimizer update rejected.")
    optimizer.step()


def refine(models, pairs, spectrum, cfg, log, calls) -> int:
    """Freeze the pairs and refine every K and a with Adam under the fit's loss."""
    for model, pair in zip(models, pairs):
        model.select_pair(tuple(pair))
    optimizer = torch.optim.Adam([q for m in models for q in (m.delay_gain, m.a)], lr=cfg["refine_lr"])
    for step in range(cfg["refine_steps"]):
        optimizer.zero_grad()
        prediction = torch.stack([m.time_domain_synth(cfg["n_fft"], *pair)
                                  for m, pair in zip(models, pairs)]).sum(0)
        loss = spectral_loss(prediction, spectrum, cfg["loss"])
        calls += len(models)
        loss.backward()
        checked_step(optimizer, models, step)
    log(calls, loss.detach(), pairs, models, force=True)
    return calls
