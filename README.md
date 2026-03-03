# DataMimic

[![Build Status](https://github.com/mthelm85/DataMimic.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mthelm85/DataMimic.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/mthelm85/DataMimic.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mthelm85/DataMimic.jl)

DataMimic generates a **synthetic DataFrame** that mimics the shape, column types, and statistical distributions of an input DataFrame — without exposing any of the original records.

## How it works

DataMimic fits an empirical model to your DataFrame in two stages:

1. **Marginals** — each column is profiled and an empirical distribution is fitted independently (frequency table for categorical columns, sorted empirical CDF for numeric columns).
2. **Copula** — a `BetaCopula` is fitted to the joint pseudo-observations of all numeric columns, preserving their dependence structure.

When you call `sample`, DataMimic draws from the copula to get correlated uniform values, inverts each through its marginal's quantile function, re-samples categorical columns independently, and re-injects missing values at the observed rate.

## Installation

```julia
using Pkg
Pkg.add("DataMimic")
```

## Quick start

```julia
using DataFrames, DataMimic

# Your private data
df = DataFrame(
    age    = rand(25:65, 500),
    income = randn(500) .* 15_000 .+ 55_000,
    region = rand(["North", "South", "East", "West"], 500),
    active = rand([true, false], 500),
)

# One-liner: fit and sample in a single call
syn = synthesize(df, 500)

# Or keep the model for repeated sampling
model = fit(df)
syn1  = sample(model, 200)
syn2  = sample(model, 1_000)

# Scramble sensitive identifier columns so real values never appear verbatim
df2 = DataFrame(
    ein    = ["12-3456789", "98-7654321"],   # tax IDs
    amount = [1200.0, 850.0],
)
syn3 = synthesize(df2, 100; scramble=[:ein])
```

## API

### `fit(df; scramble=Symbol[]) -> SynthModel`

Profiles the input DataFrame and returns a fitted `SynthModel`. No synthetic data is produced at this stage.

```julia
model = fit(df)

# Mark sensitive columns for scrambling
model = fit(df; scramble=[:ssn, :case_id])
```

Columns listed in `scramble` are sampled normally, but before being returned their values are randomly rearranged character-by-character (strings) or digit-by-digit (integers), ensuring no original identifier ever appears verbatim in the synthetic output.

### `sample(model, nrows) -> DataFrame`

Draws `nrows` synthetic observations from a fitted `SynthModel`. The output has the same column names and compatible eltypes as the original DataFrame. `nrows` does not need to equal the number of rows in the original data.

```julia
syn = sample(model, 1_000)
```

### `synthesize(df, nrows; scramble=Symbol[]) -> DataFrame`

Convenience wrapper equivalent to `sample(fit(df; scramble=scramble), nrows)`.

```julia
syn = synthesize(df, 500)
syn = synthesize(df, 500; scramble=[:ein, :case_id])
```

### `SynthModel`

The struct returned by `fit`. You can inspect it directly:

| Field | Type | Description |
|---|---|---|
| `column_names` | `Vector{Symbol}` | Ordered column names from the input |
| `column_types` | `Vector{Symbol}` | Detected type per column (see below) |
| `marginals` | `Dict{Symbol, Any}` | Fitted marginal per column |
| `missingness` | `Dict{Symbol, Float64}` | Observed missingness rate per column |
| `copula` | `BetaCopula` or `Nothing` | Fitted copula, or `Nothing` if skipped |
| `copula_columns` | `Vector{Symbol}` | Columns included in the copula |
| `nrows_original` | `Int` | Row count of the original input |
| `scrambled` | `Vector{Symbol}` | Columns whose sampled values are scrambled before output |

## Column type detection

DataMimic automatically classifies each column before fitting:

| Detected type | Criteria |
|---|---|
| `:continuous` | `Float64`/`Float32` eltype with at least one non-integer value |
| `:integer` | `Int` eltype, or float column where every value is a whole number |
| `:categorical` | `String`, `Symbol`, `Bool`, or `CategoricalArray` eltype |
| `:binary` | Any column with exactly 2 unique non-missing values |
| `:constant` | Any column with exactly 1 unique non-missing value (or all missing) |

`Union{T, Missing}` eltypes are handled correctly for all types above.

## Missing value handling

- The missingness rate of each column is recorded at fit time.
- All marginal and copula fitting operates on non-missing values only.
- During synthesis, missing values are re-injected independently per column at the recorded rate via Bernoulli draws.

## Output guarantees

- Same column names and column order as the input.
- Compatible eltypes: numeric columns preserve their original numeric type; columns with missings return `Union{T, Missing}`.
- Categorical columns only ever contain levels seen in the original data.
- Integer columns contain only integer-valued entries.
- Constant columns reproduce the single original value exactly.

## Warnings and edge cases

DataMimic emits informative warnings rather than erroring in the following situations:

| Situation | Behaviour |
|---|---|
| Only one numeric column | Copula is skipped; column sampled from its marginal directly |
| No numeric columns | Copula is skipped; all columns sampled independently |
| Column is entirely missing | Treated as a constant column with value `missing` |
| `nrows > 10 × original` | Warning that empirical marginals will repeat observed values |
| Scrambled column is `:constant` | Warning that scrambling a single-value column may have no effect |
| Scrambled column is `:continuous` | Warning that digit-scrambling floats may produce unexpected values |

## Dependencies

- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl)
- [Copulas.jl](https://github.com/lrnv/Copulas.jl)
- [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl)
