# Privacy

Three of the four engines offer formal (ε, δ)-differential privacy. They take a
[`PrivacyBudget`](@ref); the public engine rejects one rather than accepting it
and quietly providing no guarantee.

```julia
budget = PrivacyBudget(epsilon = 1.0, delta = 1e-5)
model  = fit(MSTGenerator(privacy = budget), df)
```

Every fitted private model records the budget it was trained under, so a
model that outlives the session that produced it can still say what guarantee
it carries:

```julia
model = fit(MSTGenerator(ε = 1.0), df)
privacy_budget(model)      # PrivacyBudget(ε = 1.0, δ = 1.0e-5)
```

It is shown when the model is displayed, and survives `save`/`load`.
`privacy_budget` returns `nothing` for a model from a public generator.

## What the guarantee means

Differential privacy bounds how much the output can depend on any single
record. Concretely: had one person been removed from your table before fitting,
the probability of producing any particular synthetic dataset would change by
at most a factor of about `exp(ε)` — with the `δ` term allowing a small
probability of exceeding that.

The practical consequence is that nobody can confidently determine whether a
given individual was in the input, no matter what auxiliary information they
bring. That is a strong and unusual property: it holds against attackers you
did not anticipate, and it does not decay as other datasets are published.

Smaller ε means more privacy and less utility. `δ` defaults to `1e-5` and
should stay well below `1/n`, since a δ on the order of `1/n` permits
mechanisms that expose a whole record outright.

## What the budget covers

Each private engine spends its entire budget internally, composing the pieces
using zero-concentrated differential privacy (zCDP) and converting to
(ε, δ)-DP at the end. zCDP is used because it composes tightly across many
mechanisms — the naive ε-per-step accounting would be far more pessimistic.

- **[`MSTGenerator`](@ref)** splits the budget three ways: selecting the
  spanning tree, measuring the one-way marginals, and measuring the selected
  two-way marginals. Reconciling those measurements is post-processing and
  costs nothing further.
- **[`DPCopulaGenerator`](@ref)** splits between the histogram marginals and
  the private covariance matrix.
- **[`DiffusionGenerator`](@ref)`(privacy = budget)`** trains with DP-SGD —
  per-example gradient clipping plus Gaussian noise — accounted with Rényi DP
  over Poisson-subsampled minibatches. Per-example clipping is done by ghost
  clipping, which gets each example's gradient norm without a backward pass
  per example; see [DiffusionGenerator](engines.md#DiffusionGenerator).

Sampling from a fitted private model is post-processing: draw as many synthetic
rows as you like without spending anything more. This is a genuine property of
differential privacy, not a shortcut — once the model is private, no amount of
querying it can un-privatize it.

!!! note "The reported ε is an upper bound"
    The DP-SGD accountant searches Rényi orders over a finite integer grid and
    converts to (ε, δ) with the standard bound. The true privacy loss is
    therefore no worse than reported, and typically somewhat better.

## Choosing ε

There is no universally correct value, and published deployments span a wide
range. As a rough orientation, ε below 1 is conservative, 1–3 is a common
working range, and above 8 offers weak formal protection whatever the empirical
results look like.

Rather than adopting a convention, measure what a budget costs on your data:

```julia
privacy_utility_sweep(
    MSTGenerator, df, [0.5, 1.0, 2.0, 4.0, 8.0],
    (real, synth) -> fidelity_score(real, synth).aggregate,
)
```

Two things make this harder than it looks. Results at low ε are noisy — on
Adult at ε = 0.5 the train-on-synthetic utility ratio has a seed-to-seed
standard deviation of about 0.06, larger than most differences worth acting on
— so compare distributions over several seeds rather than two single runs. And
the answer depends on row count: at a fixed ε, a private engine improves
substantially as rows increase, because its noise is fixed while the signal
grows. A budget that is unusable on ten thousand rows may be comfortable on a
hundred thousand.

## What differential privacy does not give you

A budget bounds what can be inferred about any single record's presence. It
does not make synthetic data unconditionally safe to publish, and several
things sit outside the guarantee:

- **Aggregate disclosure.** DP says nothing about whether the *patterns* you
  release are sensitive. A faithfully reproduced correlation can be exactly
  what you did not want to publish.
- **Group privacy.** The guarantee is per-record. Protection for a family, a
  household, or anyone contributing several rows degrades roughly in
  proportion to how many rows they contribute.
- **Correctness of the input.** The bound concerns the fitting procedure. If
  the table already contained data you had no right to hold, DP does not
  address that.

[`privacy_dcr`](@ref) is a useful sanity check on how close synthetic rows sit
to real ones, but it measures one sample rather than bounding all of them.
Treat it as a smoke test: a good DCR does not substitute for a formal budget,
and a formal budget does not require a good DCR.
