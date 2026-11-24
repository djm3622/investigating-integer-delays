"""Loaders for the MATLAB-generated comb (pluck) and triangle targets and their labels."""

import json
from pathlib import Path

from scipy.io import loadmat
import torch
import torchaudio


def _field(path: Path, label: str) -> str:
    fields = path.stem.split("_")
    return fields[fields.index(label) + 1]


def load_comb(path: Path) -> tuple[torch.Tensor, int, dict]:
    """Target without its leading zero, sample rate, and labels from the file name."""
    waveform, rate = torchaudio.load(path)
    metadata = loadmat(path.with_suffix(".mat"), variable_names=["pluck_delay", "impulse_gain", "scale"])
    position = int(_field(path, "d"))
    if int(metadata["pluck_delay"].item()) != position:
        raise ValueError(f"Pluck delay disagrees with filename: {path}")
    if float(metadata["impulse_gain"].item()) != 1.0 or float(metadata["scale"].item()) != 1.0:
        raise ValueError(f"Expected a unit impulse and an unscaled target: {path}")
    return waveform.squeeze(0)[1:], rate, {"K": float(_field(path, "gain")), "a": float(_field(path, "a")),
                                           "L": int(_field(path, "L")), "position": position}


def load_triangle(path: Path) -> tuple[torch.Tensor, int, dict]:
    """Target without its leading zero and with its stored scale undone, sample rate, and labels."""
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest["leading_zero"] != 1 or manifest["sign"] != "negative":
        raise ValueError("Expected one leading zero and negative excitation sign.")
    labels = loadmat(path.with_suffix(".mat"))
    scale = float(labels["scale"].item())
    if float(labels["impulse_gain"].item()) != 1.0 or not scale > 0:
        raise ValueError(f"Expected a unit impulse and a positive scale: {path}")
    waveform, rate = torchaudio.load(path)
    if rate != int(manifest["sample_rate"]) or waveform.shape[-1] != int(manifest["causal_samples"]) + 1:
        raise ValueError(f"Target disagrees with the manifest: {path}")
    waveform = waveform / scale
    return waveform[0, 1:].contiguous(), rate, {
        key: float(labels[field].item())
        for key, field in (("K", "delay_gain"), ("a", "a"), ("L", "L"), ("position", "A"))}


LOADERS = {"comb": load_comb, "triangle": load_triangle}


def load_case(kind: str, paths: list[Path], n_fft: int) -> tuple[torch.Tensor, list[dict], int]:
    """Stems (S, samples), labels of each source, and the sample rate."""
    examples = [LOADERS[kind](path) for path in paths]
    stems, rates = [e[0] for e in examples], {e[1] for e in examples}
    if len(rates) != 1 or len({s.numel() for s in stems}) != 1:
        raise ValueError("Mixture stems must have equal sample rates and lengths.")
    if any(s.numel() < n_fft or not torch.isfinite(s).all() for s in stems):
        raise ValueError("Targets must be finite and at least n_fft samples long.")
    return torch.stack(stems), [e[2] for e in examples], rates.pop()
