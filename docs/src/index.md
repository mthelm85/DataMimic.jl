# DataMimic.jl

Synthetic tabular data generation for Julia, with optional differential privacy.

DataMimic fits a generative model to a table and samples new rows preserving its
statistical structure without copying real records. Four engines sit behind one
`fit` / `sample` interface, alongside an evaluation suite for checking that the
output is actually usable.

Any [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible table works as
input, and `sample` returns the same concrete table type you passed in.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/mthelm85/DataMimic.jl")
```

[`DiffusionGenerator`](@ref) lives in a package extension; load `Lux` and
`Zygote` to activate it. For GPU training additionally load `LuxCUDA` (or
`Metal.jl` / `AMDGPU.jl`) — the device is detected at runtime and DataMimic
takes no GPU dependency itself.

## Quick start

```julia
using DataFrames, DataMimic

df = DataFrame(
    age    = rand(25:65, 500),
    income = randn(500) .* 15_000 .+ 55_000,
    region = rand(["North", "South", "East", "West"], 500),
)

model = fit(AutoGenerator(), df)
syn   = sample(model, 500)
```

With a privacy budget:

```julia
budget = PrivacyBudget(epsilon = 1.0, delta = 1e-5)
model  = fit(MSTGenerator(), df; privacy = budget)
syn    = sample(model, 500)
```

## Where to go next

- [Engines](engines.md) — what each generator does and when to reach for it
- [Privacy](privacy.md) — budgets, composition, and what is guaranteed
- [Evaluation](evaluation.md) — measuring fidelity, utility, and disclosure risk
- [API](api.md) — full reference
