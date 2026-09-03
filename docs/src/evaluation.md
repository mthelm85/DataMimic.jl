# Evaluation

A synthetic table that looks plausible can still be useless for the analysis
you intend to run on it, and one that scores well on distributional similarity
can still sit uncomfortably close to real records. DataMimic ships three
families of metric to check for each of those separately:

- **Fidelity** — does the synthetic data have the same distributions?
- **Utility** — does a model trained on it work on real data?
- **Privacy** — how close do synthetic rows sit to real ones?

They can disagree, and the disagreements are informative. High fidelity with
low utility usually means the marginals are right but the relationships between
columns are not.

## Fidelity

```julia
f = fidelity_score(real, synth)

f.aggregate             # overall distance; 0 is a perfect match
f.column_scores         # per-column distances
f.correlation_score     # disagreement between correlation matrices
f.correlation_columns   # columns included in the correlation term
f.correlation_excluded  # numeric columns skipped for having no variance
```

Every score here is a **distance**, so lower is better and zero is perfect.
Numeric columns are compared by the Kolmogorov–Smirnov statistic — the largest
gap between the two empirical CDFs — and categorical columns by total variation
distance, half the sum of absolute differences in level probabilities. Both are
bounded in [0, 1], which makes them commensurable enough to aggregate.

The correlation term compares Spearman (rank) correlation matrices. Rank
correlation is used rather than Pearson because it is invariant to the marginal
shapes already scored by the per-column terms, so the two parts measure
different things.

Constant columns are excluded from the correlation term and reported in
`correlation_excluded`: their ranks never vary, so any correlation involving
them is undefined. They are still scored individually.

Two related measures:

```julia
jensen_shannon(real, synth; n_bins = 50)      # symmetric, bounded divergence
pairwise_marginal_error(real, synth; order = 2)
```

`pairwise_marginal_error` discretizes the columns and compares every pair's
joint distribution, which catches dependence failures that per-column scores
cannot see. `order = 3` compares triples.

## Utility

```julia
t = utility_tstr(real, synth, :target; test = holdout)

t.task           # :classification or :regression, detected from the target
t.synth_score    # score for the model trained on synthetic data
t.real_score     # score for the model trained on real data
t.ratio          # synth / real
```

This is the train-on-synthetic, test-on-real protocol. Gradient-boosted trees
are fitted twice — once on synthetic data, once on real — and both are scored
against the same held-out real data. Classification is scored by macro-averaged
F1, regression by RMSE.

The ratio is the number to look at. A ratio near 1 means a model trained on
synthetic data performs about as well as one trained on the real thing, which
is usually the question that matters. Pass `test` explicitly when you have a
designated holdout; otherwise an internal split is made.

Because it is measured through a downstream task, utility is the most direct
evidence that a synthetic table is fit for purpose — and the most sensitive to
dependence structure, which is exactly what generators find hardest.

## Privacy

```julia
d = privacy_dcr(real, synth)

d.median         # median distance from a synthetic row to its nearest real row
d.exact_matches  # synthetic rows identical to some real row
```

Distance to closest record. `exact_matches` should be zero, or very near it;
anything else means the generator is reproducing records. A very small median
DCR means synthetic rows sit close to real ones, which is worth investigating
even when the engine carries a formal guarantee.

Treat this as a smoke test, not a guarantee. A good DCR does not substitute for
a privacy budget — it is an empirical observation about one sample, not a bound
over all possible datasets — and a formal budget does not require a good DCR.
See [What differential privacy does not give you](privacy.md#What-differential-privacy-does-not-give-you).

Note that DCR is O(n_synth × n_real): on large tables, evaluate on a subsample.

## Comparing engines

```julia
compare([CopulaGenerator(), CopulaGenerator(:gaussian), MSTGenerator(ε = 1.0)], df;
        metrics = (fidelity = fidelity_score,
                   utility  = utility_tstr(:income)),
        n_seeds = 5)
```

One row per generator and metric, carrying the mean and standard deviation
across seeds and the mean fit time. The result is a Tables.jl table, so
`DataFrame(…)` will sort and pivot it. Public and private generators can appear
in one call: the budget is passed only to the engines that take one.

Three things worth knowing:

**A failing engine does not stop the run.** It is reported with `ok = false`
and its error message, and the others continue. A diffusion model that diverges
overnight should not cost you the comparison of everything else.

**Read the spread, not just the mean.** On Adult at ε = 0.5, `MSTGenerator`'s
utility ratio has a seed-to-seed standard deviation of about 0.06 — larger than
most differences worth acting on. Three seeds is the floor; below that the
reported `sd` is meaningless, and `compare` warns.

**Results describe the configuration you passed, not the engine in general.**
Engines differ enormously in how much they depend on their hyperparameters. A
`DiffusionGenerator` left at the default epoch count is undertrained, and a
comparison that includes it is measuring that fact rather than the method. Give
each engine a fair configuration before drawing conclusions from the ranking.

`compare` never fits on a subsample of your table, however large it is.
Engines respond differently to row count — a private engine at fixed ε improves
substantially with more rows — so fitting on a subsample can reverse the very
ranking the comparison exists to establish. Subsample deliberately before
calling if you want the speed, and read the result as being about that smaller
dataset.

## Sweeping the budget

```julia
privacy_utility_sweep(generator, table, epsilons, metric_fn; kw...)
```

Fits the generator once per ε and applies `metric_fn(real, synth)` to each
result, which is the practical way to choose a budget for a particular dataset
rather than adopting a convention:

```julia
privacy_utility_sweep(
    MSTGenerator(), df, [0.5, 1.0, 2.0, 4.0, 8.0],
    (real, synth) -> fidelity_score(real, synth).aggregate,
)
```

The curve is usually steep at the low end and flat above it, and the useful
question is where your data stops improving. Results at low ε carry substantial
seed noise, so sweep several seeds before reading much into a single curve.
