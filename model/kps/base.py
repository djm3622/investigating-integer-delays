"""Karplus--Strong loop with learnable gain K and all-pass coefficient a on an integer (L, p) grid."""

import math

import torch
import torchaudio
from torch import nn


class KarplusStrong(nn.Module):
    """K in (gain_min, gain_max) and a in (0, 1) are sigmoids of learned scalars; (L, p) is an integer pair.

    The loop filter is (a/2 + (a + 1)/2 z^-1 + 1/2 z^-2) / (1 + a z^-1): a two-tap average and an all-pass.
    Valid pairs satisfy p < L. Subclasses supply the excitation that drives the loop.
    """

    def __init__(self, L_min: int, L_max: int, position_min: int, position_max: int,
                 gain_min: float = 0.0, gain_max: float = 1.0,
                 initial_K: float = 0.9, initial_a: float = 0.5) -> None:
        super().__init__()
        if L_min < 2 or L_max < L_min:
            raise ValueError("Invalid L bounds.")
        if position_min < 1 or position_max < position_min or position_min >= L_max:
            raise ValueError("Invalid position bounds.")
        if not 0 <= gain_min < initial_K < gain_max <= 1:
            raise ValueError("Initial gain must be strictly inside its bounds.")
        self.L_min, self.L_max = L_min, L_max
        self.position_min, self.position_max = position_min, position_max
        self.gain_min, self.gain_max = float(gain_min), float(gain_max)
        self.delay_gain = nn.Parameter(torch.logit(torch.tensor(
            (initial_K - gain_min) / (gain_max - gain_min), dtype=torch.get_default_dtype())))
        self.a = nn.Parameter(torch.logit(torch.tensor(initial_a, dtype=torch.get_default_dtype())))

    def scaled_gain(self) -> torch.Tensor:
        unit_gain = torch.sigmoid(self.delay_gain)
        if (self.gain_min, self.gain_max) == (0.0, 1.0):
            return unit_gain
        gain = self.gain_min + (self.gain_max - self.gain_min) * unit_gain
        lower, upper = gain.new_tensor(self.gain_min), gain.new_tensor(self.gain_max)
        # Sigmoid can round to an endpoint at extreme logits; keep the
        # physical gain strictly within the requested open interval.
        return gain.clamp(min=torch.nextafter(lower, upper), max=torch.nextafter(upper, lower))

    def scaled_allpass(self) -> torch.Tensor:
        return torch.sigmoid(self.a)

    def gains(self) -> tuple[float, float]:
        """K and a as numbers."""
        return float(self.scaled_gain().detach()), float(self.scaled_allpass().detach())

    @torch.no_grad()
    def set_gains(self, K: float, a: float) -> None:
        unit = min(max((K - self.gain_min) / (self.gain_max - self.gain_min), 1e-4), 1 - 1e-4)
        self.delay_gain.copy_(torch.logit(self.delay_gain.new_tensor(unit)))
        self.a.copy_(torch.logit(self.a.new_tensor(min(max(a, 1e-4), 1 - 1e-4))))

    @torch.no_grad()
    def clamp_gains(self, K_max: float, a_max: float) -> None:
        """Keep K <= K_max and a <= a_max; the lower bounds come from the scaling."""
        unit = (K_max - self.gain_min) / (self.gain_max - self.gain_min)
        self.delay_gain.clamp_(max=math.log(unit / (1 - unit)))
        self.a.clamp_(max=math.log(a_max / (1 - a_max)))

    def is_valid(self, pair: tuple[int, int]) -> bool:
        L, p = pair
        return self.L_min <= L <= self.L_max and self.position_min <= p <= self.position_max and p < L

    def excitation(self, n_samples: int, L: int, p: int, K: torch.Tensor) -> torch.Tensor:
        """Input of the loop for the pair (L, p)."""
        raise NotImplementedError

    def time_domain_synth(self, n_samples: int, L: int, p: int) -> torch.Tensor:
        """Causal response at the integer pair (L, p)."""
        if not self.is_valid((L, p)):
            raise ValueError("Expected a valid candidate pair.")
        K, denominator, b = self._integer_filter(L)
        return torchaudio.functional.lfilter(self.excitation(n_samples, L, p, K), denominator, b, clamp=False)

    def _loop_numerator(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K, a = self.scaled_gain(), self.scaled_allpass()
        return K, a, torch.stack((a / 2, (a + 1) / 2, a * 0 + 0.5))

    def _integer_filter(self, L: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """K and the (denominator, numerator) of the loop at integer delay L."""
        K, a, numerator = self._loop_numerator()
        b = numerator.new_zeros(L + 3)
        b[:3] = numerator
        denominator = numerator.new_zeros(L + 3)
        denominator[0] = 1
        denominator[1] = a
        denominator[L:L + 3] -= K * numerator
        return K, denominator, b
