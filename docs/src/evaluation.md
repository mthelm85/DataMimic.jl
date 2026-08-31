# Evaluation

Synthetic data is worth exactly what it preserves, so measure it. DataMimic
ships three families of metric: fidelity (does it look like the original),
utility (is it useful downstream), and privacy (how close does it sit to real
records).

## Fidelity

```julia
f = fidelity_score(real, synth)
f.aggregate            # overall, 0 = perfect
f.column_scores        # per column
f.correlation_score    # Spearman correlation agreement
f.correlation_excluded # numeric columns skipped (no variance)
```

Numeric columns are compared by the Kolmogorov–Smirnov statistic, categoricals
by total variation distance. Columns with no variance are excluded from the
correlation term — their ranks are constant, so any correlation involving them
is undefined — but they are still scored individually.

Related:

```julia
jensen_shannon(real, synth; n_bins = 50)
pairwise_marginal_error(real, synth; order = 2)
```

## Utility

```julia
t = utility_tstr(real, synth, :target; test = holdout)
t.synth_score   # macro-F1 training on synthetic
t.real_score    # macro-F1 training on real
t.ratio         # synth / real
```

Train-on-synthetic, test-on-real. Gradient-boosted trees are fitted to the
synthetic data and scored against real held-out data. Pass `test` explicitly
where you have a holdout; otherwise an internal split is used.

A ratio near 1 means a model trained on synthetic data is about as good as one
trained on the real thing.

## Privacy

```julia
d = privacy_dcr(real, synth)
d.median        # median distance to closest real record
d.exact_matches # synthetic rows identical to a real row
```

`exact_matches` should be zero or near it. A very small median DCR means
synthetic rows sit close to real ones, which is worth investigating even when
the generator carries a formal privacy guarantee.

## Comparing engines

Rather than guessing which engine suits your data, measure it:

```julia
compare([CopulaGenerator(), CopulaGenerator(:gaussian), MSTGenerator()], df;
        metrics = (fidelity = fidelity_score,
                   utility  = (r, s) -> utility_tstr(r, s, :income).ratio),
        n_seeds = 5,
        privacy = PrivacyBudget(epsilon = 1.0))
```

One row per generator and metric, with the mean and standard deviation across
seeds and the mean fit time. The result is a Tables.jl table, so `DataFrame(…)`
sorts and pivots it. Public and private generators can appear in the same call:
the budget is passed only to those that need it.

Three things worth knowing:

**A failing engine does not stop the run.** It is reported with `ok = false`
and its error message, and the others continue. A diffusion model that diverges
should not cost you an overnight comparison of everything else.

**Report the spread, not just the mean.** `MSTGenerator`'s utility ratio has a
seed-to-seed standard deviation near 0.06 at ε = 0.5 — larger than most
differences worth acting on. Three seeds is the floor; below that the reported
`sd` is meaningless and `compare` warns.

**Results are about the settings you passed, not the engine in the abstract.**
Engines differ enormously in how much they depend on their hyperparameters. A
`DiffusionGenerator` left at a low epoch count can score no better than
independent sampling, which says the model was undertrained, not that diffusion
is a poor method. Give each engine a fair configuration before concluding
anything from the ranking.

## Sweeping the budget

```julia
privacy_utility_sweep(generator, table, epsilons, metric_fn; kw...)
```

Fits the generator once per ε and applies `metric_fn(real, synth)`, which is the
practical way to choose a budget for a given dataset. Results at low ε carry
substantial seed noise — sweep several seeds before drawing conclusions.
