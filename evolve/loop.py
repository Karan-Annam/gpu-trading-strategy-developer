"""Evolution loop: population of strategy ASTs, GPU-swept fitness, tournament
selection. Usable as a library (the Lab drives it on a thread) or as a CLI:

    python -m evolve.loop --symbol BTCUSDT --bars 200000 --pop 32 --gens 15
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl.compiler import compile_source                # noqa: E402
from evolve import genome, sweep                       # noqa: E402

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
SEED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "strategies")

COMPLEXITY_PENALTY = 0.002     # per bytecode op
MIN_TRADES_PER_FOLD = 5
NO_TRADE_PENALTY = 2.0


class Individual:
    def __init__(self, strat, src: str):
        self.strat = strat            # param-free AST (genome)
        self.src = src                # parameterized, compilable source
        self.fitness: float | None = None
        self.stats: dict = {}

    def to_json(self) -> dict:
        return {"src": self.src, "fitness": self.fitness, **self.stats}


def evaluate(ind: Individual, bars, n_param_samples: int, n_folds: int,
             rng_seed: int, prefer_gpu: bool = True) -> None:
    prog = compile_source(ind.src)
    pm = sweep.lhs_sample(prog, n_param_samples, seed=rng_seed)
    if prog.params:
        defaults = np.asarray([prog.param_defaults()], dtype=np.float32)
        pm = np.vstack([defaults, pm])
    validation = sweep.validation_metrics(
        prog, bars, pm, n_folds=n_folds, prefer_gpu=prefer_gpu)
    scores = sweep.robust_score(validation)                 # (N,)
    if not bool(np.all(np.isfinite(scores))):
        raise ValueError("strategy produced non-finite validation scores")
    q75 = float(np.quantile(scores, 0.75))
    best_row = int(np.argmax(scores))
    best_trades = validation[:, best_row, sweep.COL["n_trades"]]
    med_trades = float(np.median(best_trades))
    min_trades = float(np.min(best_trades))
    fitness = q75 - COMPLEXITY_PENALTY * len(prog.code)
    if min_trades < MIN_TRADES_PER_FOLD:
        fitness -= NO_TRADE_PENALTY
    ind.fitness = float(fitness)
    ind.stats = {
        "validation_q75_score": q75,
        "best_validation_row_score": float(scores[best_row]),
        "best_params": {p["name"]: float(pm[best_row, j])
                        for j, p in enumerate(prog.params)},
        "median_validation_trades": med_trades,
        "min_validation_trades": min_trades,
        "mean_validation_sharpe": float(
            validation[:, :, sweep.COL["sharpe"]].mean()),
        "n_ops": len(prog.code),
    }


def evaluate_holdout(ind: Individual, bars, prefer_gpu: bool = True) -> None:
    """Evaluate the selected program/parameters once on untouched final bars."""
    prog = compile_source(ind.src)
    params = np.asarray([[ind.stats["best_params"][p["name"]]
                          for p in prog.params]], dtype=np.float32)
    metrics = sweep.pick_engine(prefer_gpu).run_batch(prog, bars, params)[0]
    ind.stats["holdout"] = {
        name: float(metrics[column]) for name, column in sweep.COL.items()
    }


def seed_population(rng: random.Random, pop_size: int) -> list[Individual]:
    pop: list[Individual] = []
    for path in sorted(glob.glob(os.path.join(SEED_DIR, "*.dsl"))):
        with open(path, encoding="utf-8") as f:
            strat = genome.from_source(f.read())
        src = genome.to_source(strat)
        if src:
            pop.append(Individual(strat, src))
    while len(pop) < pop_size:
        strat, src = genome.spawn_valid(lambda: genome.random_strategy(rng), rng)
        if strat is not None:
            pop.append(Individual(strat, src))
    return pop[:pop_size]


def next_generation(pop: list[Individual], rng: random.Random, elite: int = 2,
                    p_crossover: float = 0.4,
                    target_size: int | None = None) -> list[Individual]:
    evaluated = [ind for ind in pop if ind.fitness is not None]
    if not evaluated:
        raise ValueError("next_generation requires evaluated individuals")
    target_size = len(pop) if target_size is None else target_size
    if target_size < 1:
        raise ValueError("target_size must be positive")
    ranked = sorted(evaluated, key=lambda i: i.fitness, reverse=True)
    elite = min(elite, target_size, len(ranked))
    out = [Individual(ranked[i].strat, ranked[i].src) for i in range(elite)]
    for i in range(elite):                     # elites keep their evaluation
        out[i].fitness = ranked[i].fitness
        out[i].stats = ranked[i].stats

    def tournament() -> Individual:
        return max(rng.sample(evaluated, k=min(3, len(evaluated))),
                   key=lambda i: i.fitness)

    while len(out) < target_size:
        if rng.random() < p_crossover and len(evaluated) >= 2:
            child = genome.crossover(tournament().strat, tournament().strat, rng)
        else:
            child = genome.mutate(tournament().strat, rng)
        strat, src = genome.spawn_valid(lambda c=child: genome.mutate(c, rng)
                                        if rng.random() < 0.3 else c, rng)
        if strat is None:
            strat, src = genome.spawn_valid(lambda: genome.random_strategy(rng), rng)
        if strat is not None:
            out.append(Individual(strat, src))
    return out


class EvolutionRun:
    """Owns one run; the Lab polls .status while a thread executes run()."""

    def __init__(self, bars, pop_size=32, generations=15, n_param_samples=48,
                 n_folds=3, seed=0, prefer_gpu=True, inject=None, label="",
                 holdout_frac=0.2, run_root=RUNS_DIR):
        if not 0.0 < holdout_frac < 0.5:
            raise ValueError("holdout_frac must be between 0 and 0.5")
        split = int(len(bars) * (1.0 - holdout_frac))
        if split < 100 or len(bars) - split < 20:
            raise ValueError("not enough bars for search and final holdout")
        if pop_size < 2 or generations < 1 or n_param_samples < 1 or n_folds < 1:
            raise ValueError("invalid evolution resource counts")
        self.bars = bars.slice(0, split)
        self.holdout_bars = bars.slice(split, len(bars))
        self.pop_size = pop_size
        self.generations = generations
        self.n_param_samples = n_param_samples
        self.n_folds = n_folds
        self.rng = random.Random(seed)
        self.prefer_gpu = prefer_gpu
        self.inject = inject           # callable(run, gen) -> list[Individual]
        self.stop_requested = False
        self.gen = 0
        self.population: list[Individual] = []
        self.history: list[dict] = []
        self.state = "idle"
        self.error: str | None = None
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            run_root, f"run_{ts}_{uuid.uuid4().hex[:8]}"
            f"{('_' + label) if label else ''}")

    # ---- status for the Lab ----
    def status(self) -> dict:
        best = self.best()
        return {
            "state": self.state,
            "generation": self.gen,
            "generations": self.generations,
            "pop_size": self.pop_size,
            "history": self.history,
            "best": best.to_json() if best else None,
            "error": self.error,
            "run_dir": os.path.basename(self.run_dir),
            "search_bars": len(self.bars),
            "holdout_bars": len(self.holdout_bars),
        }

    def best(self) -> Individual | None:
        evald = [i for i in self.population if i.fitness is not None]
        return max(evald, key=lambda i: i.fitness) if evald else None

    def population_json(self) -> list[dict]:
        ranked = sorted((i for i in self.population if i.fitness is not None),
                        key=lambda i: i.fitness, reverse=True)
        return [i.to_json() for i in ranked]

    # ---- main loop ----
    def run(self) -> None:
        try:
            self.state = "running"
            os.makedirs(self.run_dir, exist_ok=True)
            self.population = seed_population(self.rng, self.pop_size)
            for g in range(self.generations):
                if self.stop_requested:
                    break
                self.gen = g + 1
                for k, ind in enumerate(self.population):
                    if ind.fitness is None:
                        evaluate(ind, self.bars, self.n_param_samples,
                                 self.n_folds, rng_seed=g * 1000 + k,
                                 prefer_gpu=self.prefer_gpu)
                    if self.stop_requested:
                        break
                best = self.best()
                self.history.append({
                    "gen": self.gen,
                    "best_fitness": best.fitness if best else None,
                    "mean_fitness": float(np.mean(
                        [i.fitness for i in self.population if i.fitness is not None])),
                })
                self._save_gen()
                if self.stop_requested or g == self.generations - 1:
                    break
                if self.inject is not None:
                    for j, ind in enumerate(self.inject(self, g) or []):
                        try:
                            evaluate(ind, self.bars, self.n_param_samples,
                                     self.n_folds,
                                     rng_seed=(g + 1) * 100_000 + j,
                                     prefer_gpu=self.prefer_gpu)
                        except Exception:
                            continue  # an invalid proposal must not abort a run
                        self.population.append(ind)
                self.population = next_generation(
                    self.population, self.rng, target_size=self.pop_size)
            best = self.best()
            if best is not None and not self.stop_requested:
                evaluate_holdout(best, self.holdout_bars, self.prefer_gpu)
                self._save_gen()  # overwrite the final snapshot with holdout metrics
            self.state = "stopped" if self.stop_requested else "done"
        except Exception as e:              # surfaced through /api/evolve/status
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"

    def _save_gen(self) -> None:
        path = os.path.join(self.run_dir, f"gen_{self.gen:03d}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.population_json(), f, indent=1)
        os.replace(tmp, path)
        best = self.best()
        if best:
            best_path = os.path.join(self.run_dir, "best.dsl")
            best_tmp = best_path + ".tmp"
            with open(best_tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(best.src)
            os.replace(best_tmp, best_path)


def main() -> None:
    from data.ohlcv import load_bars
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--bars", type=int, default=200_000)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bars = load_bars(args.symbol)
    bars = bars.slice(len(bars.close) - args.bars, len(bars.close))
    run = EvolutionRun(bars, pop_size=args.pop, generations=args.gens,
                       n_param_samples=args.samples, n_folds=args.folds,
                       seed=args.seed)
    t0 = time.time()

    import threading
    th = threading.Thread(target=run.run)
    th.start()
    printed = 0
    while th.is_alive() or printed < len(run.history):
        if th.is_alive():
            th.join(timeout=2.0)
        while printed < len(run.history):
            h = run.history[printed]
            print(f"gen {h['gen']:3d}  best={h['best_fitness']:8.3f}  "
                  f"mean={h['mean_fitness']:8.3f}  ({time.time() - t0:.0f}s)")
            printed += 1
    best = run.best()
    if best:
        holdout = best.stats.get("holdout", {})
        print(f"\nbest fitness {best.fitness:.3f} "
              f"(validation sharpe {best.stats['mean_validation_sharpe']:.2f}, "
              f"holdout sharpe {holdout.get('sharpe', float('nan')):.2f})\n")
        print(best.src)
        print(f"saved -> {run.run_dir}")


if __name__ == "__main__":
    main()
