"""Continuous loop/position delays synthesized with Lagrange fractional delays.

The interpolator is the fifth-order centered Lagrange FIR, adapted from the
diffKS implementation (github.com/ptablasdpaula/diffKS_torchLPC). Integer
estimates are decoded by rounding both delays.
"""

import torch
from torch import nn
from torch.nn import functional as F

LAGRANGE_ORDER = 5


def lagrange_delay_fir(delay: torch.Tensor, size: int) -> torch.Tensor:
    """Fifth-order centered Lagrange FIR, shifted at the causal left boundary.

    At an integer delay the taps are exactly a unit impulse at that delay.
    """
    nodes = torch.arange(LAGRANGE_ORDER + 1, dtype=delay.dtype, device=delay.device)
    start = max(0, int(delay.detach().floor()) - LAGRANGE_ORDER // 2)
    u = delay - start
    weights = torch.stack([
        torch.prod((u - nodes[nodes != k]) / (k - nodes[nodes != k]))
        for k in range(LAGRANGE_ORDER + 1)
    ])
    return F.pad(weights, (start, size - start - LAGRANGE_ORDER - 1))


def delayed_loop_filter(numerator: torch.Tensor, delay: torch.Tensor,
                        size: int) -> torch.Tensor:
    """Convolve the three-tap loop numerator with a fractional delay."""
    fir = lagrange_delay_fir(delay, size)
    return sum(numerator[i] * F.pad(fir, (i, 0))[:size] for i in range(3))


class LagrangeDelayPairMixin:
    """Continuous L and position as parameters, clamped to the grid with round(p) < round(L)."""

    def __init__(self, *args, L_init: float, position_init: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delay_len_parameter = nn.Parameter(self.a.new_tensor(L_init))
        self.position_parameter = nn.Parameter(self.a.new_tensor(position_init))

    def continuous_L(self) -> torch.Tensor:
        return self.delay_len_parameter.clamp(
            max(float(self.L_min), float(self.position_min) + 1), float(self.L_max))

    def continuous_position(self) -> torch.Tensor:
        upper = self.continuous_L().detach().round() - 1
        position = self.position_parameter.clamp(float(self.position_min), float(self.position_max))
        return torch.minimum(position, upper)

    def _loop_coefficients(self, L: torch.Tensor, size: int):
        K, a, numerator = self._loop_numerator()
        denominator = (F.pad(torch.stack((a * 0 + 1, a)), (0, size - 2))
                       - K * delayed_loop_filter(numerator, L, size))
        return K, numerator, denominator
