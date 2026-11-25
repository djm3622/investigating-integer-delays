"""The letter's test cases: which dataset files make up each mixture, and the target noise."""

import hashlib
from pathlib import Path
import random

import torch

ROOT = Path(__file__).resolve().parents[1]


def make_cases(config: dict) -> list[dict]:
    """One row per excitation, source count and instance.

    Separate deterministic draws for each source count; no stem repeats within a
    source-count stratum. The draw covers instances_drawn instances, of which the
    first `instances` are used.
    """
    cases = []
    for kind, directory in config["datasets"].items():
        paths = sorted((ROOT / directory).glob("*.wav"))
        drawn = config["instances_drawn"]
        if len(paths) < drawn * max(config["source_counts"]):
            raise ValueError(f"{directory} has too few WAVs.")
        for count in config["source_counts"]:
            generator = random.Random(f"{config['selection_seed']}:{kind}:{count}")
            chosen = generator.sample(range(len(paths)), drawn * count)
            for instance in range(config["instances"]):
                indices = sorted(chosen[instance * count:(instance + 1) * count])
                cases.append({"filter": kind, "source_count": count, "instance": instance,
                              "paths": [paths[i] for i in indices]})
    return cases


def add_target_noise(target: torch.Tensor, case: dict, snr_db: float, config: dict) -> torch.Tensor:
    """White Gaussian noise at snr_db over the fitted first n_fft samples.

    The realization depends only on the case, so every loss sees the same noise.
    """
    key = f"{config['selection_seed']}:{case['filter']}:{case['source_count']}:{case['instance']}:noise"
    generator = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
    window = config["fit"]["n_fft"]
    signal_power = target[:window].square().mean()
    noise = noise * (signal_power / 10 ** (snr_db / 10)).sqrt() / noise[:window].square().mean().sqrt()
    return target + noise
