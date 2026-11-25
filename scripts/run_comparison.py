"""Run the letter's loss comparison: one JSON summary per fit, skipping fits already done."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import yaml

from data.cases import ROOT, add_target_noise, make_cases
from data.dataset import load_case
from fitting.gradient_descent import fit_gradient_descent
from fitting.losses import LOSSES
from model.kps import INTEGER_MODELS

SOLVER = "gd"  # names each fit's directory and its summary


def noise_label(snr_db) -> str:
    return "clean" if snr_db is None else f"snr{snr_db:g}"


def make_jobs(config: dict) -> list[dict]:
    """Every fit of the comparison, with its output path."""
    return [{**case, "loss": loss, "solver": SOLVER, "snr_db": snr_db, "seed": config["seed"],
             "run_id": (f"{noise_label(snr_db)}_{loss}/{case['filter']}/n{case['source_count']:02}/"
                        f"i{case['instance']:03}/{SOLVER}")}
            for loss in config["losses"] for snr_db in config["snr_db"] for case in make_cases(config)]


def select_jobs(jobs: list[dict], config: dict, args: argparse.Namespace) -> list[dict]:
    """The jobs kept by the command-line options; an option left out keeps the configuration's choice."""
    return [job for job in jobs
            if job["loss"] in (args.losses or config["losses"])
            and job["filter"] in (args.filters or config["datasets"])
            and job["source_count"] in (args.counts or config["source_counts"])
            and job["instance"] < (args.instances or config["instances"])]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_target(job: dict, config: dict) -> tuple[torch.Tensor, list[dict]]:
    """The mixture of the job's stems, with its noise if any, and the labels of every source."""
    stems, truth, _ = load_case(job["filter"], job["paths"], config["fit"]["n_fft"])
    target = stems.sum(0)
    if job["snr_db"] is not None:
        target = add_target_noise(target, job, job["snr_db"], config)
    return target, truth


def run_fit(job: dict, config: dict, root: Path) -> dict:
    torch.set_num_threads(1)
    output = root / job["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text())
    seed_everything(job["seed"])
    target, truth = load_target(job, config)
    progress = output / "progress.jsonl"
    progress.write_text("")
    start = time.perf_counter()
    fit = fit_gradient_descent(job["filter"], job["source_count"], target, job["loss"], config["model"],
                               config["fit"], progress)
    summary = {**{k: v for k, v in job.items() if k != "paths"}, "files": [p.name for p in job["paths"]],
               "fit_seconds": time.perf_counter() - start, "source_syntheses": fit.syntheses,
               "found": [list(pair) for pair in fit.pairs], "found_K": fit.K, "found_a": fit.a,
               "true": truth, "history": fit.history}
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def safe_fit(job: dict, config: dict, root: Path) -> tuple[str, float | None, str | None]:
    try:
        return job["run_id"], run_fit(job, config, root)["fit_seconds"], None
    except Exception:
        return job["run_id"], None, traceback.format_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/loss_comparison.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "output/loss_comparison")
    parser.add_argument("--losses", nargs="+", choices=LOSSES)
    parser.add_argument("--filters", nargs="+", choices=tuple(INTEGER_MODELS))
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--instances", type=int, help="First instances of every case")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    jobs = select_jobs(make_jobs(config), config, args)
    print(f"{len(jobs)} fits", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(safe_fit, job, config, args.output.resolve()) for job in jobs]
        for future in as_completed(futures):
            print(*(value for value in future.result() if value is not None), flush=True)


if __name__ == "__main__":
    main()
