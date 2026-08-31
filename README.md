<p align="center">
  <img src="docs/src/assets/logo-wordmark.png" width="320"
       alt="DataMimic.jl: a mimic octopus holding a bar chart, a pie chart
            and a data grid, above the DataMimic.jl wordmark">
</p>

# DataMimic.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mthelm85.github.io/DataMimic.jl/dev/)
[![CI](https://github.com/mthelm85/DataMimic.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mthelm85/DataMimic.jl/actions/workflows/CI.yml)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a.svg)](https://github.com/aviatesk/JET.jl)

Synthetic tabular data generation for Julia, with optional differential privacy.

DataMimic fits a generative model to a table and samples new rows that preserve
its statistical structure without copying real records. It ships four engines —
from a fast copula to a differentially private diffusion model — behind one
`fit` / `sample` interface, and an evaluation suite for checking that the
output is actually any good.

Any [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible table works
as input, and `sample` returns the same concrete table type you passed in.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/mthelm85/DataMimic.jl")
```

`DiffusionGenerator` is provided by a package extension. To use it, load
[Lux.jl](https://github.com/LuxDL/Lux.jl) and
[Zygote.jl](https://github.com/FluxML/Zygote.jl) as well:

```julia
Pkg.add(["Lux", "Zygote"])
```

For GPU training, additionally load `LuxCUDA` (or `Metal.jl` / `AMDGPU.jl`) —
DataMimic detects the device at runtime and takes no GPU dependency itself.

## Quick start

```julia
using DataFrames, DataMimic

df = DataFrame(
    age    = rand(25:65, 500),
    income = randn(500) .* 15_000 .+ 55_000,
    region = rand(["North", "South", "East", "West"], 500),
    active = rand([true, false], 500),
)

model = fit(CopulaGenerator(), df)
syn   = sample(model, 500)

# Or fit and sample in one call
syn = synthesize(CopulaGenerator(), df, 500)
```

### With differential privacy

Private engines require a `PrivacyBudget`; public engines reject one.

```julia
budget = PrivacyBudget(epsilon = 1.0, delta = 1e-5)

model = fit(MSTGenerator(), df; privacy = budget)
syn   = sample(model, 500)
```

### Excluding identifiers

Identifier columns are kept out of the statistical model entirely, so real
values never reach the synthetic table. Give each one a `fill` spec to
regenerate it on output — **without a fill spec the column is dropped from the
result**.

```julia
df = DataFrame(
    ein     = ["12-3456789", "98-7654321", "55-1122334"],
    amount  = [1200.0, 850.0, 2310.0],
    quarter = ["Q1", "Q2", "Q1"],
)

model = fit(CopulaGenerator(), df;
            identifiers = [:ein],
            fill        = Dict(:ein => :sequential))
syn = sample(model, 100)     # ein = "ein_1", "ein_2", ...
```

A fill spec is one of:

| Spec | Result |
|---|---|
| `:sequential` | `"<colname>_1"`, `"<colname>_2"`, … |
| `:sequential_int` | `1`, `2`, `3`, … |
| a `String` prefix | `"prefix_1"`, `"prefix_2"`, … |
| a `Function` | `f(i)` for row `i` |

## Engines

| Generator | Private | Notes |
|---|---|---|
| `CopulaGenerator(:beta \| :gaussian)` | no | Fast, and strong on mixed tables. Copula over numeric *and* categorical columns |
| `DiffusionGenerator(; dp = false)` | optional | TabDDPM. Highest fidelity; `dp = true` enables DP-SGD |
| `MSTGenerator(2)` | yes | MST with Private-PGM reconciliation. Good on categorical-heavy data |
| `DPCopulaGenerator()` | yes | DP histogram marginals + Analyze-Gauss private covariance |

### Choosing one

Which engine wins depends on the table, and not in ways that are reliable to
predict from its shape: engines that rank one way on one dataset routinely
swap on another. Rather than guess, measure — [`compare`](#comparing-engines)
fits several engines to *your* data and scores them.

### Class-conditional diffusion

For a classification-style table, naming the label column conditions the model
on it. This substantially improves downstream utility, and reproduces the
TabDDPM paper's setup.

```julia
using Lux, Zygote   # activates the extension

gen = DiffusionGenerator(
    epochs        = 3750,
    batch_size    = 4096,
    d_layers      = [256, 1024, 1024, 1024, 1024, 256],
    num_timesteps = 100,
    target        = :income_bracket,
)
model = fit(gen, df)
syn   = sample(model, nrow(df))
```

## API

### Fitting and sampling

```julia
fit(generator, table; privacy = nothing, hints = ColumnHint[],
                      identifiers = Symbol[], fill = Dict(),
                      rng = Random.default_rng())
sample(model, n; rng = model.rng)
synthesize(generator, table, n; kw...)
```

- `privacy::Union{Nothing, PrivacyBudget}` — required by private generators
- `hints::Vector{ColumnHint}` — override column type detection
- `identifiers::Vector{Symbol}` — columns to exclude from the model
- `fill` — how to repopulate identifier columns on output
- `rng` — stored on the model, so sampling is reproducible

### Types

```julia
PrivacyBudget(; epsilon, delta = 1e-5)
ColumnHint(; name, kind, levels = nothing)
```

Valid `kind` values: `:continuous`, `:integer`, `:categorical`, `:binary`,
`:constant`, `:identifier`.

### Persistence

```julia
save(path, model)
model = load(path)
```

Uses Julia's `Serialization`, so files are portable within a Julia version but
may not load across versions. A version header is written and checked.

### Comparing engines

```julia
compare([CopulaGenerator(), MSTGenerator()], df;
        metrics = (fidelity = fidelity_score,
                   utility  = (r, s) -> utility_tstr(r, s, :income).ratio),
        n_seeds = 5,
        privacy = PrivacyBudget(epsilon = 1.0))
```

Fits each generator, samples, and scores it — one row per generator/metric with
mean, standard deviation across seeds, and fit time. A failing engine is
reported rather than aborting the run. The result is a Tables.jl table.

### Evaluation

```julia
fidelity_score(real, synth)                       # marginal + correlation agreement
privacy_dcr(real, synth)                          # distance to closest record
utility_tstr(real, synth, target; test = nothing) # train-on-synthetic, test-on-real
jensen_shannon(real, synth; n_bins = 50)
pairwise_marginal_error(real, synth; order = 2)
privacy_utility_sweep(generator, table, epsilons, metric_fn; kw...)
```

`fidelity_score` returns per-column scores plus an aggregate, all in `[0, 1]`
where `0` is perfect. Numeric columns with no variance are excluded from the
correlation term and reported in `correlation_excluded`.

`utility_tstr` trains gradient-boosted trees on the synthetic data and scores
them on real held-out data, reporting macro-F1 and a synthetic/real ratio.

## Column type detection

Columns are classified as `:continuous`, `:integer`, `:categorical`,
`:binary`, `:constant`, or `:identifier`. Detection is cardinality-aware:
low-cardinality integers in a large sample are treated as categorical, and
high-cardinality ones as integer. Pass a `ColumnHint` to override.

## Missing values

Missingness is measured per column at fit time and reintroduced at the same
rate when sampling, so the synthetic table has a comparable missingness
profile. Columns that are entirely missing are treated as constant.

## Reproducibility

Pass `rng` to `fit` and it is stored on the model; `sample` uses it unless you
pass a different one. Fitting twice with equal seeds gives identical models.

## References

Engines follow their published algorithms, cross-checked against the reference
implementations rather than the papers alone. See
[`references/REFERENCES.md`](references/REFERENCES.md) for the full list and
for the specific places where this package deviates.

## License

MIT
