"""mcmc_pt: parallel-tempering search over every source's (L, p, K, a)."""

import math

import torch
from torch import nn

from fitting.common import Log, refine, set_a, set_K
from model.kps.losses import spectral_loss


def fit_mcmc_pt(models: nn.ModuleList, target, cfg, log_path=None):
    """Metropolis chains targeting exp(-loss / T); returns (history, charged syntheses).

    Cached source waves make a one-source proposal cost one synthesis. The search stops at
    search_syntheses * S charged syntheses; the lowest-loss state found starts the K/a refinement.
    """
    n = cfg["n_fft"]
    spectrum = torch.fft.rfft(target[:n])
    grids = [model.all_pairs() for model in models]
    valid = [set(grid) for grid in grids]
    sources, temperatures = len(models), cfg["temperatures"]
    budget = cfg["search_syntheses"] * sources
    log = Log(log_path, cfg["log_syntheses"])
    calls = sweep = drawn = noop = accepted = exchanges = 0

    def uniform(i):
        return grids[i][int(torch.randint(len(grids[i]), ()))]

    def wave(i, pair, K, a):
        set_K(models[i], K)
        set_a(models[i], a)
        return models[i].time_domain_synth(n, *pair)

    def score(waves):
        return float(spectral_loss(sum(waves), spectrum, cfg["loss"]))

    with torch.no_grad():
        chains = []
        for pairs in [[uniform(i) for i in range(sources)] for _ in temperatures]:
            K = [cfg["initial_K"]] * sources
            a = [cfg["initial_a"]] * sources
            waves = [wave(i, pairs[i], K[i], a[i]) for i in range(sources)]
            calls += sources
            chains.append({"pairs": pairs, "K": K, "a": a, "waves": waves, "loss": score(waves)})
        best = min(chains, key=lambda c: c["loss"])
        best = {key: list(best[key]) for key in ("pairs", "K", "a")} | {"loss": best["loss"]}
        while calls < budget:
            for chain, T in zip(chains, temperatures):
                drawn += 1
                move = float(torch.rand(()))
                i = int(torch.randint(sources, ()))
                changed = {}
                if move < cfg["swap_move"] and sources > 1:
                    # Exchange L, position or both between two sources.
                    j = int(torch.randint(sources - 1, ()))
                    j += j >= i
                    kind = int(torch.randint(3, ()))
                    p, q = chain["pairs"][i], chain["pairs"][j]
                    new_i = (q[0] if kind != 1 else p[0], q[1] if kind != 0 else p[1])
                    new_j = (p[0] if kind != 1 else q[0], p[1] if kind != 0 else q[1])
                    if new_i in valid[i] and new_j in valid[j]:
                        changed = {i: (new_i, chain["K"][i], chain["a"][i]),
                                   j: (new_j, chain["K"][j], chain["a"][j])}
                elif move < cfg["swap_move"] + cfg["continuous_move"]:
                    K = chain["K"][i] + cfg["K_step"] * float(torch.randn(()))
                    a = chain["a"][i] + cfg["a_step"] * float(torch.randn(()))
                    if models[i].gain_min < K < models[i].gain_max and 0 < a < 1:
                        changed = {i: (chain["pairs"][i], K, a)}
                else:
                    # A +/- local_radius step with probability local_probability, else a uniform redraw.
                    if float(torch.rand(())) < cfg["local_probability"]:
                        step = torch.randint(-cfg["local_radius"], cfg["local_radius"] + 1, (2,)).tolist()
                        pair = (chain["pairs"][i][0] + step[0], chain["pairs"][i][1] + step[1])
                    else:
                        pair = uniform(i)
                    if pair in valid[i] and pair != chain["pairs"][i]:
                        changed = {i: (pair, chain["K"][i], chain["a"][i])}
                if not changed:
                    noop += 1
                    continue
                waves = list(chain["waves"])
                for index, (pair, K, a) in changed.items():
                    waves[index] = wave(index, pair, K, a)
                calls += len(changed)
                loss = score(waves)
                if loss <= chain["loss"] or float(torch.rand(())) < math.exp((chain["loss"] - loss) / T):
                    accepted += 1
                    for index, (pair, K, a) in changed.items():
                        chain["pairs"][index], chain["K"][index], chain["a"][index] = pair, K, a
                    chain["waves"], chain["loss"] = waves, loss
                    if loss < best["loss"]:
                        best = {"pairs": list(chain["pairs"]), "K": list(chain["K"]),
                                "a": list(chain["a"]), "loss": loss}
            sweep += 1
            if sweep % cfg["exchange_every"] == 0:
                for c in range(len(chains) - 1):
                    delta = (1 / temperatures[c] - 1 / temperatures[c + 1]) * (
                        chains[c]["loss"] - chains[c + 1]["loss"])
                    if delta >= 0 or float(torch.rand(())) < math.exp(delta):
                        chains[c], chains[c + 1] = chains[c + 1], chains[c]
                        exchanges += 1
            log(calls, best["loss"], best["pairs"], models)
        for i in range(sources):
            set_K(models[i], best["K"][i])
            set_a(models[i], best["a"][i])
    log.rows.append({"calls": calls, "proposals_drawn": drawn, "proposals_noop": noop,
                     "proposals_accepted": accepted, "sweeps": sweep, "exchanges_accepted": exchanges,
                     "temperatures": list(temperatures), "search_budget": budget})
    return log.rows, refine(models, best["pairs"], spectrum, cfg, log, calls)
