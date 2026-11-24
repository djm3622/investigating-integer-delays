"""Causal Karplus--Strong model with a two-tap pluck-position comb."""

import torch
import torchaudio
from torch.nn import functional as F

from .base import KarplusStrong
from .fractional_delay import LagrangeDelayPairMixin, delayed_loop_filter


class KarplusStrongPluck(KarplusStrong):
    """Loop delay L and pluck delay p from a unit impulse.

    The MATLAB generator uses input -K * (delta[n] - delta[n-p]) and feedback
    +K * y[n-L].
    """

    def time_domain_synth(self, n_samples: int, L: int, p: int) -> torch.Tensor:
        if not self.is_valid((L, p)):
            raise ValueError("Expected a valid candidate pair.")
        K, denominator, b = self._integer_filter(L)
        impulse = denominator.new_zeros(n_samples)
        impulse[0] = 1
        if p < n_samples:
            impulse[p] -= 1
        return torchaudio.functional.lfilter(-K * impulse, denominator, b, clamp=False)


class KarplusStrongPluckLagrange(LagrangeDelayPairMixin, KarplusStrongPluck):
    """Continuous L and p; causal synthesis with Lagrange fractional delays."""

    def __init__(self, *args, L_init: float, position_init: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_continuous_pair(L_init, position_init)

    def lagrange_synth(self, n_samples: int) -> torch.Tensor:
        L, p = self.continuous_L(), self.continuous_position()
        size = int(torch.maximum(L, p).detach().floor()) + 8
        K, numerator, denominator = self._loop_coefficients(L, size)
        b = -K * (F.pad(numerator, (0, size - 3)) - delayed_loop_filter(numerator, p, size))
        impulse = L.new_zeros(n_samples)
        impulse[0] = 1
        return torchaudio.functional.lfilter(impulse, denominator, b, clamp=False)
