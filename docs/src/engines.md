# Engines

| Generator | Private | Reach for it when |
|---|---|---|
| [`CopulaGenerator`](@ref) | no | You want speed. Strong on mixed numeric/categorical tables |
| [`DiffusionGenerator`](@ref) | optional | You want the highest fidelity, and can afford training time |
| [`MSTGenerator`](@ref) | yes | Private, categorical-heavy data |
| [`DPCopulaGenerator`](@ref) | yes | Private, continuous-heavy data |

## Choosing between them

The table above says what each engine is *for*, which is enough to narrow the
field but not to pick a winner. Relative performance depends on the data in
ways that table shape does not predict — the ordering that holds on one dataset
frequently reverses on another, and for private engines it also moves with ε
and with row count.

So measure instead of guessing. [`compare`](@ref) fits a list of engines to
your own table, repeats each over several seeds, and reports the mean and
spread of whatever metrics you name. See
[Comparing engines](evaluation.md#Comparing-engines).

## CopulaGenerator

A copula over the modelled columns, with empirical marginals. `:beta` fits a
`BetaCopula`; `:gaussian` fits a Gaussian copula.

Categorical and binary columns take part in the copula through an ordinal
encoding of their empirical CDF (the *distributional transform*): a level
occupying `[F(k-1), F(k)]` maps to a uniform draw inside that interval, and
sampling inverts the same CDF. Dependence between categorical and numeric
columns is therefore modelled rather than discarded.

Two caveats. The association a copula can express is monotone in the level
order, which is arbitrary for a nominal variable — so this captures a real part
of the dependence, not all of it. And `:beta` handles this far better than
`:gaussian`, because it is nonparametric and can represent the non-monotone
structure an arbitrary ordering produces. On Adult, train-on-synthetic utility
is 0.99 for `:beta` against 0.66 for `:gaussian`; prefer the default unless you
have a reason not to.

A categorical column with only one level cannot be encoded, so it is left out
of the copula and drawn independently.

## DiffusionGenerator

TabDDPM: Gaussian diffusion on numeric features and multinomial diffusion on
categoricals, with a plain MLP denoiser. Requires `Lux` and `Zygote`.

Naming a `target` column enables class-conditional generation, which
substantially improves downstream utility on classification-style tables:

```julia
using Lux, Zygote

gen = DiffusionGenerator(
    epochs        = 3750,
    batch_size    = 4096,
    d_layers      = [256, 1024, 1024, 1024, 1024, 256],
    num_timesteps = 100,
    target        = :income_bracket,
)
model = fit(gen, df)
```

Setting `dp = true` trains with DP-SGD and requires a `PrivacyBudget`.

## MSTGenerator

Discretizes every column, measures all one-way marginals under Gaussian noise,
selects a spanning tree over columns with the exponential mechanism, measures
the selected two-way marginals, then reconciles every measurement into a
consistent tree-structured model before sampling.

The reconciliation step matters most at tight budgets, where measurement noise
dominates; by `ε ≈ 4` it makes little difference.

## DPCopulaGenerator

Differentially private histogram marginals combined with a private covariance
matrix (Analyze-Gauss), assembled into a Gaussian copula. Suits
continuous-heavy tables at moderate ε.
