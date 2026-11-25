"""Feedback-delay effects that follow a string model: echo, Schroeder comb, damped comb, all-pass.

Each effect has its own delay D and loop gain g:

echo / Schroeder comb  y[n] = x[n] + g y[n - D]                     H = 1 / (1 - g z^-D)
damped comb            the loop also holds a one-pole lowpass        H = 1 / (1 - g (1 - d) z^-D / (1 - d z^-1))
Schroeder all-pass     y[n] = -g x[n] + x[n - D] + g y[n - D]        H = (z^-D - g) / (1 - g z^-D)

An echo and a Schroeder comb are the same filter; they differ only in the range of D and g they use.
A cascade string -> effect has the response H_string(L, p) * H_effect(D).
"""

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from model.kps.fractional_delay import lagrange_delay_fir

EFFECTS = ("echo", "schroeder_comb", "damped_comb", "allpass")


class FeedbackDelay(nn.Module):
    """One feedback-delay effect with a fixed gain g and damping d; D is an integer or a continuous parameter."""

    def __init__(self, kind: str, gain: float, damping: float = 0.0, D_init: float | None = None) -> None:
        super().__init__()
        if kind not in EFFECTS:
            raise ValueError(f"Unknown effect: {kind}")
        if not 0 <= gain < 1 or not 0 <= damping < 1:
            raise ValueError("Require 0 <= g < 1 and 0 <= d < 1.")
        if kind != "damped_comb" and damping:
            raise ValueError("Only the damped comb has a loop lowpass.")
        self.kind, self.gain, self.damping = kind, float(gain), float(damping)
        self.delay_parameter = None if D_init is None else nn.Parameter(torch.tensor(float(D_init)))

    def _coefficients(self, delay_fir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(numerator, denominator) for a delay given as an FIR (unit impulse at D, or Lagrange taps)."""
        size = delay_fir.numel()
        one = F.pad(delay_fir.new_ones(1), (0, size - 1))
        g, d = self.gain, self.damping
        if self.kind == "allpass":
            return delay_fir - g * one, one - g * delay_fir
        if self.kind == "damped_comb":
            lowpass = one - d * F.pad(one, (1, 0))[:size]
            return lowpass, lowpass - g * (1 - d) * delay_fir
        return one, one - g * delay_fir

    def _filter(self, x: torch.Tensor, delay_fir: torch.Tensor) -> torch.Tensor:
        numerator, denominator = self._coefficients(delay_fir)
        return torchaudio.functional.lfilter(x, denominator, numerator, clamp=False)

    def time_domain(self, x: torch.Tensor, D: int) -> torch.Tensor:
        """Filter x through the effect at integer delay D >= 1."""
        if D < 1:
            raise ValueError("Require D >= 1.")
        delay_fir = x.new_zeros(D + 2)
        delay_fir[D] = 1
        return self._filter(x, delay_fir)

    def continuous_D(self) -> torch.Tensor:
        if self.delay_parameter is None:
            raise ValueError("Construct the effect with D_init for a continuous delay.")
        return self.delay_parameter.clamp(min=3.0)

    def lagrange(self, x: torch.Tensor) -> torch.Tensor:
        """Filter x through the effect at the continuous delay, with a fifth-order Lagrange FIR."""
        D = self.continuous_D()
        return self._filter(x, lagrange_delay_fir(D, int(D.detach().floor()) + 8))

    def response(self, n_bins: int, D: float | torch.Tensor) -> torch.Tensor:
        """Frequency response on n_bins points of [0, pi], with an ideal (fractional) delay."""
        omega = torch.linspace(0, torch.pi, n_bins, dtype=torch.float64)
        z_D = torch.exp(-1j * omega * torch.as_tensor(D, dtype=torch.float64))
        g, d = self.gain, self.damping
        if self.kind == "allpass":
            return (z_D - g) / (1 - g * z_D)
        loop = (1 - d) / (1 - d * torch.exp(-1j * omega)) if self.kind == "damped_comb" else 1.0
        return 1 / (1 - g * loop * z_D)


class StringWithEffect(nn.Module):
    """A comb or triangle string model followed by a feedback-delay effect."""

    def __init__(self, string: nn.Module, effect: FeedbackDelay) -> None:
        super().__init__()
        self.string, self.effect = string, effect

    def time_domain_synth(self, n_samples: int, L: int, p: int, D: int) -> torch.Tensor:
        return self.effect.time_domain(self.string.time_domain_synth(n_samples, L, p), D)

    def lagrange_synth(self, n_samples: int) -> torch.Tensor:
        """Continuous L, p (string) and D (effect); needs Lagrange variants of both."""
        return self.effect.lagrange(self.string.lagrange_synth(n_samples))
