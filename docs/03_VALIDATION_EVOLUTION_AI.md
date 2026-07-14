# Validation, evolution, and AI integration

## Sweep statistics

`evolve/sweep.py` builds full grids or Latin-hypercube samples and evaluates
contiguous rolling train/test folds. `walk_forward_select` chooses a row on each
training slice and reports the following slice as out-of-sample for that fold.
The Lab labels these fold results as validation. It reports the simplified
deflated Sharpe score only for a single-period sweep, because applying that
formula to averaged, overlapping fold Sharpes would be misleading.

`robust_score` is:

```text
mean(validation Sharpe) - 0.5 * std(validation Sharpe)
```

The deflated Sharpe implementation uses the normal-return form, annualized
Sharpe values rescaled per bar, and the number of attempted configurations. It
does not currently include measured skew and kurtosis.

## Evolution search

An evolution run reserves the final 20% of its selected chronological bars as a
holdout. The first 80% is search data. Within that search portion, rolling fold
test slices are repeatedly used for fitness and are therefore called
validation, not a final untouched test set.

For each individual the engine:

1. Compiles the strategy.
2. Samples its parameter region with Latin hypercube sampling plus defaults.
3. Scores each row across validation folds.
4. Uses the 75th percentile row score, minus bytecode complexity and low-trade
   penalties, as fitness.

Genomes store literal ASTs. A deterministic pass promotes selected windows,
thresholds, and risk values into parameters before evaluation. Mutation and
crossover operate on typed strategy components, and the compiler rejects
invalid offspring.

After the last generation, the selected program and selected parameter row are
run once on the held-out final 20%. That holdout result is reported separately
and is not fed back into selection.

Generated run checkpoints live under `evolve/runs/` and are ignored by Git.

## Claude integration

When `ANTHROPIC_API_KEY` is set, `evolve/llm.py` supports:

- generating DSL from a short strategy idea;
- suggesting a revision to the current DSL;
- injecting a novel candidate into an evolution run.

The model name can be overridden with `BT_LLM_MODEL`. Model responses are
treated as untrusted DSL source: they pass through the same lexer, parser,
compiler, window bounds, stack limits, and state limits as handwritten source.
A failed compilation gets one repair attempt. The returned text is not executed
as Python, JavaScript, or shell code.

This boundary protects the host application from generated code execution. It
does not make a generated trading rule profitable or eliminate statistical
overfitting.

## Lab

The Lab is a FastAPI backend with a local browser UI. It provides compile
errors, backtest charts, a trade table, parameter heatmaps, walk-forward
metrics, evolution progress, and an accept/reject diff for model suggestions.
Large chart series and trade markers are decimated/capped before being sent to
the browser.
