"""Spectral losses of the letter, between a synthesized window and the target's rfft.

log     L1 between 10 log10 magnitudes
lin     L1 between linear magnitudes, over the mean target magnitude
tilt    lin with both magnitudes weighted by 1 / (omega^2 + omega_c^2)
mr_lin  auraloss multi-resolution STFT loss, linear magnitude term only
mr_log  auraloss multi-resolution STFT loss, log magnitude term only
"""

from math import pi

import torch

EPS = 1e-7
TILT_CUTOFF = 2 * pi / 400  # rad/sample
# auraloss defaults: FFT 1024/2048/512, hops 120/240/50, Hann windows 600/1200/240.
MULTI_RESOLUTION = {"mr_lin": dict(w_sc=0.0, w_log_mag=0.0, w_lin_mag=1.0),
                    "mr_log": dict(w_sc=0.0, w_log_mag=1.0, w_lin_mag=0.0)}
LOSSES = ("log", "lin", "tilt", *MULTI_RESOLUTION)
_multi_resolution = {}


def log_magnitude(spectrum: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(spectrum.abs() + EPS)


def tilt_weight(target_spectrum: torch.Tensor) -> torch.Tensor:
    omega = torch.linspace(0, pi, target_spectrum.numel(), dtype=target_spectrum.real.dtype,
                           device=target_spectrum.device)
    return 1 / (omega.square() + TILT_CUTOFF ** 2)


def spectral_loss(prediction: torch.Tensor, target_spectrum: torch.Tensor, name: str) -> torch.Tensor:
    if name in MULTI_RESOLUTION:
        if name not in _multi_resolution:
            from auraloss.freq import MultiResolutionSTFTLoss
            _multi_resolution[name] = MultiResolutionSTFTLoss(**MULTI_RESOLUTION[name])
        # The target window is recovered exactly from its rfft.
        target = torch.fft.irfft(target_spectrum, n=prediction.shape[-1])
        return _multi_resolution[name](prediction[None, None], target[None, None])
    spectrum = torch.fft.rfft(prediction)
    if name == "log":
        return (log_magnitude(spectrum) - log_magnitude(target_spectrum)).abs().mean()
    if name == "lin":
        return (spectrum.abs() - target_spectrum.abs()).abs().mean() / target_spectrum.abs().mean()
    if name == "tilt":
        weight = tilt_weight(target_spectrum)
        weighted = target_spectrum.abs() * weight
        return (spectrum.abs() * weight - weighted).abs().mean() / weighted.mean()
    raise ValueError(f"Unknown loss: {name}")
