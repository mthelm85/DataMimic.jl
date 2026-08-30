# Privacy

Private engines take a [`PrivacyBudget`](@ref) and satisfy (ε, δ)-differential
privacy. Public engines reject a budget rather than silently ignoring it.

```julia
budget = PrivacyBudget(epsilon = 1.0, delta = 1e-5)
model  = fit(MSTGenerator(), df; privacy = budget)
```

`δ` defaults to `1e-5`. Smaller `ε` means more privacy and less utility.

## What the budget covers

Each private engine spends its whole budget internally and composes the pieces
using zero-concentrated differential privacy (zCDP), which converts to
(ε, δ)-DP at the end.

- **`MSTGenerator`** splits the budget across selecting the tree, measuring the
  one-way marginals, and measuring the selected two-way marginals. Reconciling
  those measurements is post-processing and costs nothing further.
- **`DPCopulaGenerator`** splits between the histogram marginals and the private
  covariance matrix.
- **`DiffusionGenerator(dp = true)`** trains with DP-SGD — per-example gradient
  clipping plus Gaussian noise — accounted with Rényi DP over Poisson-subsampled
  minibatches.

Sampling from a fitted private model is post-processing: draw as many synthetic
rows as you like without spending more budget.

## Choosing ε

There is no universally correct value. As a rough guide, ε below 1 is
conservative, 1–3 is a common working range, and above 8 offers weak formal
protection. Measure rather than guess:

```julia
using DataMimic

results = privacy_utility_sweep(
    MSTGenerator(), df, [0.5, 1.0, 2.0, 4.0, 8.0],
    (real, synth) -> fidelity_score(real, synth).aggregate,
)
```

Benchmarks at low ε are noisy. On Adult at ε = 0.5, the train-on-synthetic
utility ratio has a seed-to-seed standard deviation of about 0.06 — larger than
most differences worth acting on. Compare distributions over several seeds
rather than two single runs.

## What differential privacy does not give you

A budget bounds what can be inferred about any single record's presence. It does
not make synthetic data unconditionally safe to publish, and it says nothing
about whether the *aggregate* patterns you release are sensitive. Use
[`privacy_dcr`](@ref) as a sanity check on how close synthetic rows sit to real
ones, but treat it as a smoke test rather than a guarantee — a good DCR does not
substitute for a formal budget, and a formal budget does not require a good DCR.
