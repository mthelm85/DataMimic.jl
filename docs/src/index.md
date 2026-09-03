# DataMimic.jl

Synthetic tabular data generation for Julia, with optional differential privacy.

DataMimic fits a generative model to a table and draws new rows from it. The
synthetic rows are not copies or perturbations of real ones: they are samples
from a distribution estimated from your data, so column distributions and the
relationships between columns are preserved while individual records are not.

That makes it useful when the real table cannot be shared — for a public
release, a demo, a test fixture, or a collaborator outside the data-sharing
agreement — and when you need more rows than you have.

Four engines sit behind one `fit` / `sample` interface, three of them offering
formal (ε, δ)-differential privacy. An evaluation suite measures how well the
result holds up, because a synthetic table that looks plausible can still be
useless for the analysis you intend to run on it.

Any [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible table works
as input — `DataFrame`, `NamedTuple` of vectors, `CSV.File`, and so on — and
`sample` returns the same concrete type you passed in.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/mthelm85/DataMimic.jl")
```

[`DiffusionGenerator`](@ref) lives in a package extension. Load `Lux` and
`Zygote` alongside DataMimic to activate it; without them the other three
engines work as normal. For GPU training also load `LuxCUDA` (or `Metal.jl` /
`AMDGPU.jl`) — the device is detected at runtime, and DataMimic itself takes no
GPU dependency.

## Quick start

```julia
using DataFrames, DataMimic

df = DataFrame(
    age    = rand(25:65, 500),
    income = randn(500) .* 15_000 .+ 55_000,
    region = rand(["North", "South", "East", "West"], 500),
)

model = fit(CopulaGenerator(), df)
syn   = sample(model, 500)
```

`fit` inspects the table, classifies each column, and estimates the model.
`sample` draws however many rows you ask for — more than the original if you
want them.

To fit and sample in one step:

```julia
syn = synthesize(CopulaGenerator(), df, 500)
```

### With differential privacy

Private engines require a [`PrivacyBudget`](@ref); public ones reject it rather
than accepting it and quietly providing no guarantee.

```julia
budget = PrivacyBudget(epsilon = 1.0, delta = 1e-5)
model  = fit(MSTGenerator(privacy = budget), df)
syn    = sample(model, 500)
```

Drawing samples from a fitted private model costs nothing further, however many
rows you take. See [Privacy](privacy.md).

### Choosing an engine

There is no engine that wins everywhere, and which one suits a table is not
reliably predictable from its shape. [`compare`](@ref) fits several to your own
data and reports how each did:

```julia
compare([CopulaGenerator(), CopulaGenerator(:gaussian), MSTGenerator(ε = 1.0)], df)
```

## Where to go next

- [Preparing your data](data.md) — column detection, identifiers, missing values
- [Engines](engines.md) — how each generator works, and when to reach for it
- [Privacy](privacy.md) — budgets, composition, and what is guaranteed
- [Evaluation](evaluation.md) — measuring fidelity, utility, and disclosure risk
- [API](api.md) — full reference
