# API Reference

## Fitting and sampling

```@docs
fit
sample
synthesize
```

## Generators

```@docs
CopulaGenerator
DiffusionGenerator
MSTGenerator
DPCopulaGenerator
```

## Fitted models

`fit` returns one of these, depending on the generator. Pass it to
[`sample`](@ref), or persist it with [`save`](@ref).

```@docs
FittedCopulaModel
FittedMSTModel
FittedDPCopulaModel
FittedDiffusionModel
```

## Configuration

```@docs
PrivacyBudget
ColumnHint
```

## Persistence

```@docs
save
load
```

## Evaluation

```@docs
compare
fidelity_score
utility_tstr
privacy_dcr
jensen_shannon
pairwise_marginal_error
privacy_utility_sweep
```
