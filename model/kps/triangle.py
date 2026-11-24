"""Unit-impulse triangle excitation and positive-feedback KS resonator."""

import torch
import torchaudio
from torch.nn import functional as F

from .base import KarplusStrong
from .fractional_delay import LagrangeDelayPairMixin


def triangle(n_samples: int, L: int | torch.Tensor, A: int | torch.Tensor,
             reference: torch.Tensor) -> torch.Tensor:
    """Unit-height, zero-based triangle supported on [0, L]."""
    n = torch.arange(n_samples, device=reference.device, dtype=reference.dtype)
    L = torch.as_tensor(L, device=n.device, dtype=n.dtype)
    A = torch.as_tensor(A, device=n.device, dtype=n.dtype)
    if not (bool((L >= 2).item()) and bool((A >= 1).item()) and bool((A < L).item())):
        raise ValueError("Require L >= 2 and 1 <= A < L.")
    return torch.where(n <= A, n / A, torch.clamp((L - n) / (L - A), min=0))


class KarplusStrongTriangle(KarplusStrong):
    """String pulled aside at p = A and released: the loop is driven by -K times a triangle."""

    def excitation(self, n_samples: int, L: int | torch.Tensor, A: int | torch.Tensor,
                   K: torch.Tensor) -> torch.Tensor:
        return -K * triangle(n_samples, L, A, self.a)

    def time_domain_synth(self, n_samples: int, L: int, p: int) -> torch.Tensor:
        if not self.is_valid((L, p)):
            raise ValueError("Expected a valid candidate pair.")
        K, denominator, b = self._integer_filter(L)
        output = torchaudio.functional.lfilter(self.excitation(n_samples, L, p, K), denominator, b,
                                               clamp=False)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("Nonfinite causal triangle output.")
        return output


class KarplusStrongTriangleLagrange(LagrangeDelayPairMixin, KarplusStrongTriangle):
    """Continuous L and A; causal synthesis with a Lagrange fractional loop delay."""

    def __init__(self, *args, L_init: float, position_init: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_continuous_pair(L_init, position_init)

    def lagrange_synth(self, n_samples: int) -> torch.Tensor:
        L, A = self.continuous_L(), self.continuous_position()
        size = int(L.detach().floor()) + 8
        K, numerator, denominator = self._loop_coefficients(L, size)
        output = torchaudio.functional.lfilter(
            self.excitation(n_samples, L, A, K), denominator, F.pad(numerator, (0, size - 3)), clamp=False)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("Nonfinite causal triangle output.")
        return output
