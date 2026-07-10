"""Genome operators + a miniature end-to-end evolution run."""

import random

import pytest

from dsl.compiler import compile_source
from engine import cpu_engine
from evolve import genome
from evolve.loop import EvolutionRun
from tests.util import random_walk_bars


def test_random_strategies_mostly_compile():
    rng = random.Random(0)
    ok = 0
    for _ in range(60):
        src = genome.to_source(genome.random_strategy(rng))
        if src is not None:
            ok += 1
            prog = compile_source(src)
            assert len(prog.params) <= 8
    assert ok >= 55          # the generator should almost always be valid


def test_mutation_keeps_validity():
    rng = random.Random(1)
    strat, _ = genome.spawn_valid(lambda: genome.random_strategy(rng), rng)
    ok = 0
    for _ in range(40):
        m = genome.mutate(strat, rng)
        if genome.to_source(m) is not None:
            ok += 1
    assert ok >= 30


def test_crossover_produces_valid_children():
    rng = random.Random(2)
    a, _ = genome.spawn_valid(lambda: genome.random_strategy(rng), rng)
    b, _ = genome.spawn_valid(lambda: genome.random_strategy(rng), rng)
    ok = 0
    for _ in range(20):
        child = genome.crossover(a, b, rng)
        if genome.to_source(child) is not None:
            ok += 1
    assert ok >= 16


def test_strip_and_reparameterize_seed():
    with open("strategies/sma_cross.dsl", encoding="utf-8") as f:
        src0 = f.read()
    strat = genome.from_source(src0)
    assert strat.params == []
    src = genome.to_source(strat)
    assert src is not None
    prog = compile_source(src)
    assert 1 <= len(prog.params) <= 4
    for p in prog.params:
        assert p["lo"] < p["hi"]
        assert p["lo"] <= p["default"] <= p["hi"]


@pytest.mark.skipif(not cpu_engine.available(), reason="btcpu not built")
def test_miniature_evolution_run():
    bars = random_walk_bars(2500, seed=3, drift=0.0001)
    run = EvolutionRun(bars, pop_size=6, generations=2, n_param_samples=8,
                       n_folds=2, seed=4, prefer_gpu=False)
    run.run()
    assert run.state == "done", run.error
    assert run.best() is not None
    assert run.best().fitness is not None
    assert len(run.history) == 2
    pop = run.population_json()
    assert all("src" in p and p["fitness"] is not None for p in pop)
