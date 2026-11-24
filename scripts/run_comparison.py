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
from torch import nn
import yaml

from data.cases import ROOT, add_target_noise, make_cases
from data.dataset import load_case
from fitting.gradient_descent import fit_gradient_descent
from fitting.mcmc import fit_mcmc_pt
from model.kps.losses import LOSSES
from model.kps.pluck import KarplusStrongPluck
from model.kps.triangle import KarplusStrongTriangle

MODELS = {"comb": KarplusStrongPluck, "triangle": KarplusStrongTriangle}


def noise_label(snr_db) -> str:
    return "clean" if snr_db is None else f"snr{snr_db:g}"


def make_jobs(config: dict) -> list[dict]:
    """Every fit of the comparison, with its output path."""
    return [{**case, "loss": loss, "solver": solver, "snr_db": snr_db, "seed": config["seed"],
             "run_id": (f"{noise_label(snr_db)}_{loss}/{case['filter']}/n{case['source_count']:02}/"
                        f"i{case['instance']:03}/{solver}")}
            for loss in config["losses"] for snr_db in config["snr_db"]
            for case in make_cases(config) for solver in config["solvers"]]


def run_fit(job: dict, config: dict, root: Path) -> dict:
    torch.set_num_threads(1)
    output = root / job["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text())
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    settings, cfg = config["model"], {**config["fit"], "loss": job["loss"]}
    models = nn.ModuleList(MODELS[job["filter"]](**settings) for _ in range(job["source_count"]))
    stems, truth, _ = load_case(job["filter"], job["paths"], cfg["n_fft"])
    target = stems.sum(0)
    if job["snr_db"] is not None:
        target = add_target_noise(target, job, job["snr_db"], config)
    progress = output / "progress.jsonl"
    progress.write_text("")
    start = time.perf_counter()
    if job["solver"] == "gd":
        history, calls = fit_gradient_descent(models, target, cfg, job["filter"], settings, progress)
    else:
        search = config["mcmc_pt"]
        scale = search["temperature_scale"][job["loss"]]
        if scale is None:
            raise ValueError(f"Set mcmc_pt.temperature_scale.{job['loss']} in the config.")
        cfg |= {**search, "temperatures": [T * scale for T in search["temperatures"]],
                "initial_K": settings["initial_K"], "initial_a": settings["initial_a"]}
        history, calls = fit_mcmc_pt(models, target, cfg, progress)
    summary = {**{k: v for k, v in job.items() if k != "paths"}, "files": [p.name for p in job["paths"]],
               "fit_seconds": time.perf_counter() - start, "source_syntheses": calls,
               "found": [list(m.pair) for m in models],
               "found_K": [float(m.scaled_gain().detach()) for m in models],
               "found_a": [float(m.scaled_allpass().detach()) for m in models],
               "true": truth, "history": history}
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def safe_fit(job, config, root):
    try:
        return job["run_id"], run_fit(job, config, root)["fit_seconds"], None
    except Exception:
        return job["run_id"], None, traceback.format_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/loss_comparison.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "output/loss_comparison")
    parser.add_argument("--losses", nargs="+", choices=LOSSES)
    parser.add_argument("--solvers", nargs="+", choices=("gd", "mcmc_pt"))
    parser.add_argument("--filters", nargs="+", choices=tuple(MODELS))
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--instances", type=int, help="First instances of every case")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    jobs = [job for job in make_jobs(config)
            if job["loss"] in (args.losses or config["losses"])
            and job["solver"] in (args.solvers or config["solvers"])
            and job["filter"] in (args.filters or config["datasets"])
            and job["source_count"] in (args.counts or config["source_counts"])
            and job["instance"] < (args.instances or config["instances"])]
    print(f"{len(jobs)} fits", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(safe_fit, job, config, args.output.resolve()) for job in jobs]
        for future in as_completed(futures):
            print(*(value for value in future.result() if value is not None), flush=True)


if __name__ == "__main__":
    main()
