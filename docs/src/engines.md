# Engines

| Generator | Private | Reach for it when |
|---|---|---|
| [`CopulaGenerator`](@ref) | no | You want speed and the dependence that matters is between numeric columns |
| [`DiffusionGenerator`](@ref) | optional | You want the highest fidelity, and can afford training time |
| [`MSTGenerator`](@ref) | yes | Private, categorical-heavy data |
| [`DPCopulaGenerator`](@ref) | yes | Private, continuous-heavy data |
| [`AutoGenerator`](@ref) | either | You would rather not choose |

## AutoGenerator dispatch

Let `D` be the number of modelled columns (identifiers excluded) and `N` the row
count.

**Without a privacy budget**

- `D ≤ 30` → `CopulaGenerator(:beta)`
- `D > 30` or `N > 100_000` → `DiffusionGenerator(dp = false)`

**With a privacy budget**

- `N < 20_000`, categorical fraction > 50% → `MSTGenerator(2)`
- `N < 20_000`, categorical fraction ≤ 50% → `DPCopulaGenerator()`
- `N ≥ 20_000`, `D > 30` → `DiffusionGenerator(dp = true)`
- `N ≥ 20_000`, `D ≤ 30` → `MSTGenerator(2)`

## CopulaGenerator

A copula over the numeric columns, with empirical marginals. `:beta` fits a
`BetaCopula`; `:gaussian` fits a Gaussian copula to rank-based
pseudo-observations.

Categorical and binary columns are sampled independently from their empirical
distributions and take no part in the copula, so dependence between a
categorical column and anything else is not modelled. This is the fastest
engine and a reasonable default for non-private use.

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
