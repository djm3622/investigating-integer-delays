# Investigating integer delays

Code for the letter on learning the integer delays of a recursive comb by
gradient descent: which spectral losses give the delay a slope, and how many
restarts gradient descent needs for one to seven simultaneous plucked strings.

The model is a Karplus--Strong loop with integer loop delay `L` and pluck
position `p`, loop gain `K` and all-pass coefficient `a`, driven by a two-tap
comb or a triangle excitation.

This repository holds only the code needed for the letter. It was extracted from
[DiffClarinet](https://github.com/djm3622/DiffClarinet), where the model, the
losses and the fitting were developed and which keeps the full experiment
history (branch `search-prototypes`).

## Layout

- `model/kps/`: the string models. `base.py` holds the loop (gain, all-pass,
  integer grid), `pluck.py` and `triangle.py` the comb and triangle
  excitations, `fractional_delay.py` the Lagrange relaxation to continuous
  delays.
- `model/effects.py`: feedback-delay effects with their own delay `D`
  (echo, Schroeder comb, damped comb, Schroeder all-pass), and `StringWithEffect`,
  which cascades either string model into one of them.
- `fitting/losses.py`: the spectral losses (`log`, `lin`, `tilt`, and the
  multi-resolution `mr_lin` and `mr_log`).
- `fitting/stages.py`: the stages of a fit: descent on relaxed delays, integer
  polish, and refinement of `K` and `a`.
- `fitting/gradient_descent.py`: gradient descent with restarts,
  `fit_gradient_descent`, which runs the stages and keeps the history of a fit.
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

Each fit writes `summary.json` under
`output/loss_comparison/<noise>_<loss>/<excitation>/n<sources>/i<instance>/gd/`;
finished fits are skipped, so a run can be resumed.
The scorer's bootstrap draws from one shared generator, so its intervals depend
on which fits are present: the letter's intervals were drawn from an output
tree that also held the parallel-tempering fits, and scoring a gradient-descent-only
tree gives the same point estimates with slightly different intervals.

```bash
python -m scripts.run_comparison --losses lin --filters triangle --counts 1
python -m scripts.score_comparison
python -m scripts.figures.notch
python -m scripts.figures.slope_direction
python -m scripts.figures.restart_curve
```

## Acknowledgement

Claude (Anthropic) was used to help write the experiment code.
