# Investigating integer delays

Code for the letter on learning the integer delays of a recursive comb by
gradient descent: which spectral losses give the delay a slope, and how many
restarts gradient descent needs for one to seven simultaneous plucked strings.

The model is a Karplus--Strong loop with integer loop delay `L` and pluck
position `p`, loop gain `K` and all-pass coefficient `a`, driven by a two-tap
comb or a triangle excitation.

This repository holds only the code needed for the letter. It was extracted from
[DiffClarinet](https://github.com/djm3622/DiffClarinet), where the model, the
losses and the solvers were developed and which keeps the full experiment
history (branch `search-prototypes`).

## Layout

- `model/kps/`: comb and triangle models, Lagrange fractional delays, and the
  spectral losses (`log`, `lin`, `tilt`, and the multi-resolution `mr_lin` and
  `mr_log`).
- `fitting/`: gradient descent with restarts (`gd`) and the parallel-tempering
  search (`mcmc_pt`).
- `data/`: target loaders, and the test cases (files and noise) of the letter.
- `scripts/`: the loss comparison, its scoring, and the figures.
- `configs/loss_comparison.yaml`: every setting of the comparison.

## Setup

```bash
pip install -r requirements.txt
```

Place the MATLAB-generated targets in `data/vary_all_pluck` (comb) and
`data/vary_all_triangle` (triangle). They are not part of this repository.

## Usage

Each fit writes `summary.json`; finished fits are skipped, so a run can be
resumed. The full comparison took about 4,100 CPU-hours for ten losses, of
which this repository keeps five.

```bash
python -m scripts.run_comparison --losses lin --filters triangle --counts 1 --solvers gd
python -m scripts.score_comparison
python -m scripts.figures.notch
python -m scripts.figures.slope_direction
python -m scripts.figures.restart_curve
```

`mcmc_pt` needs the temperature scale of each loss in the configuration.

## Acknowledgement

Claude (Anthropic) was used to help write the experiment code.
