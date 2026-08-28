# DataMimic.jl v2.0 — Technical Specification

## 1. Overview

DataMimic.jl v2.0 replaces the single-copula generator with a **multi-engine synthetic
data toolkit** built on `Tables.jl`. It introduces differential-privacy (DP) mechanisms,
an automated engine selector, and an evaluation suite.

Version 2.0 intentionally **breaks backward compatibility** with v1.x.

### Design Principles

| # | Principle | Consequence |
|---|-----------|-------------|
| 1 | **Tables.jl native** | Every public function accepts any `Tables.istable` source and returns the same concrete type the caller passed in (via `Tables.materializer`). |
| 2 | **Type-stable artifacts** | Generator configs and fitted models are distinct concrete types — no `Any`-typed fields. |
| 3 | **Reproducibility** | Every stochastic function accepts an `rng::AbstractRNG` keyword (default `Random.default_rng()`). |
| 4 | **Lightweight core** | Heavy dependencies (Lux.jl) live behind Julia package extensions; the core has zero deep-learning deps. |
| 5 | **Privacy by construction** | A `PrivacyBudget` is required for private generators and rejected by public ones — invalid combinations are caught at `fit` time, not silently ignored. |
| 6 | **Identifiers are excluded, not obfuscated** | Identifier columns carry zero statistical signal. They are dropped from the output by default, or filled with user-supplied placeholders — never character-shuffled. |

### Phased Delivery

| Phase | Contents | Milestone |
|-------|----------|-----------|
| **Phase 1 — Foundation** | Type system (including `PrivacyBudget` struct), Tables.jl plumbing, `CopulaGenerator` (port from v1), identifier handling, column-type detection, `AutoGenerator` (public-only dispatch), serialization | v2.0-alpha |
| **Phase 2 — Privacy** | `MSTGenerator`, `DPCopulaGenerator`, AutoGenerator private dispatch | v2.0-beta |
| **Phase 3 — Deep Generative** | `DiffusionGenerator` (Lux extension): TabDDPM with multinomial diffusion for categoricals; non-private mode first, then DP-SGD | v2.0-rc |
| **Phase 4 — Evaluation** | `DataMimic.Evaluate` submodule (fidelity, DCR, TSTR via DecisionTree.jl) | v2.0 |
| **Phase 4b — Extended Evaluation** | Jensen–Shannon divergence, pairwise marginal error, privacy–utility sweep | v2.0.1 |

**Phase 3 detail — DiffusionGenerator:**  The core TabDDPM (noise schedule,
ResNet MLP with timestep embedding, Gaussian + multinomial diffusion,
denoising loop) is ~800–1200 lines of Lux.jl. DP-SGD adds ~300–500 lines
for per-sample gradient clipping, noise injection, and a Rényi DP
accountant.

**Why Lux over Flux?**  Lux separates model structure from parameters
(`model`, `ps`, `st`), making each forward pass a pure function of `ps`.
This is critical for DP-SGD: per-sample gradients reduce to
`Zygote.gradient(ps -> loss(model, ps, st, x[i]), ps)` — no hooking into
implicit parameter mutation, no micro-batching workaround. Lux also
serializes cleanly (parameters are plain NamedTuples) and is where the
SciML ecosystem is actively investing. Start with `AutoZygote()` as the
AD backend (Lux Tier I, most tested); swap to `AutoEnzyme()` later — it's
a one-token change via Lux's Training API.

---

## 2. Type System

### 2.1 Abstract Hierarchy

```julia
abstract type AbstractGenerator end
abstract type AbstractPublicGenerator  <: AbstractGenerator end
abstract type AbstractPrivateGenerator <: AbstractGenerator end

abstract type AbstractFittedModel end
```

### 2.2 Privacy Budget

```julia
Base.@kwdef struct PrivacyBudget
    epsilon::Float64
    delta::Float64 = 1e-5

    function PrivacyBudget(epsilon, delta)
        epsilon > 0   || throw(ArgumentError("ε must be positive, got $epsilon"))
        0 ≤ delta < 1 || throw(ArgumentError("δ must be in [0, 1), got $delta"))
        new(epsilon, delta)
    end
end
```

### 2.3 Generator Configs

```julia
# Auto-selector — dispatches to a concrete generator at fit time.
struct AutoGenerator <: AbstractGenerator end

# ── Public generators ────────────────────────────────────────────────────

struct CopulaGenerator <: AbstractPublicGenerator
    copula_type::Symbol   # :beta (default) or :gaussian

    function CopulaGenerator(copula_type::Symbol)
        copula_type in (:beta, :gaussian) ||
            throw(ArgumentError("copula_type must be :beta or :gaussian, got :$copula_type"))
        new(copula_type)
    end
end
CopulaGenerator() = CopulaGenerator(:beta)

# ── Private generators ───────────────────────────────────────────────────

"""
    MSTGenerator

Private synthetic data via the MST (McKenna et al., 2021) algorithm:
select low-error 2-way marginals using the exponential mechanism, build a
junction tree, and reconstruct a full joint distribution with calibrated
Gaussian noise.

Replaces the earlier "GraphicalDPGenerator" name to cite the actual algorithm.
"""
struct MSTGenerator <: AbstractPrivateGenerator
    max_marginal_order::Int   # 2 for 2-way, 3 for 3-way (default 2)

    function MSTGenerator(max_marginal_order::Int)
        max_marginal_order in (2, 3) ||
            throw(ArgumentError("max_marginal_order must be 2 or 3, got $max_marginal_order"))
        new(max_marginal_order)
    end
end
MSTGenerator() = MSTGenerator(2)

"""
    DPCopulaGenerator

DP-noisy quantile marginals + private covariance Gaussian copula.
Suited for continuous-heavy tables under moderate ε.
"""
struct DPCopulaGenerator <: AbstractPrivateGenerator end

"""
    DiffusionGenerator

TabDDPM with optional DP-SGD. Requires the `LuxExt` package extension
(activated by `using Lux`).
"""
Base.@kwdef struct DiffusionGenerator <: AbstractGenerator
    dp::Bool        = false
    epochs::Int     = 100
    batch_size::Int = 512

    function DiffusionGenerator(dp, epochs, batch_size)
        epochs > 0     || throw(ArgumentError("epochs must be positive, got $epochs"))
        batch_size > 0 || throw(ArgumentError("batch_size must be positive, got $batch_size"))
        new(dp, epochs, batch_size)
    end
end
```

> **Note on `DiffusionGenerator`:** it subtypes `AbstractGenerator` (not Public
> or Private) because its `dp` flag determines privacy at fit time. The `fit`
> method enforces: `dp == true` requires a `PrivacyBudget`; `dp == false`
> rejects one.

### 2.4 Column Schema Hints

Users can override auto-detected column types:

```julia
Base.@kwdef struct ColumnHint
    name::Symbol
    kind::Symbol              # :continuous, :integer, :categorical, :binary, :constant, :identifier
    levels::Union{Nothing, Vector} = nothing   # lock categorical levels
end
```

The `:identifier` kind tells `fit` to skip the column during statistical
modelling entirely. See §4.3 for how identifier columns are handled in
the output.

### 2.5 Fitted-Model Types

Each generator produces its own concrete `AbstractFittedModel`:

```julia
struct FittedCopulaModel{C, M} <: AbstractFittedModel
    column_names::Vector{Symbol}
    column_kinds::Vector{Symbol}
    marginals::Dict{Symbol, Marginal}
    missingness::Dict{Symbol, Float64}
    copula::C                     # BetaCopula, GaussianCopula, or Nothing
    copula_columns::Vector{Symbol}
    n_original::Int
    identifier_columns::Vector{Symbol}
    identifier_fills::Dict{Symbol, FillSpec}
    materializer::M
    rng::AbstractRNG
end

struct FittedMSTModel <: AbstractFittedModel
    # ... MST-specific junction tree, noisy marginals, etc.
end

struct FittedDPCopulaModel <: AbstractFittedModel
    # ... DP-noisy quantiles, private covariance, etc.
end

struct FittedDiffusionModel <: AbstractFittedModel
    # ... trained Lux model, normalization params, etc.
end
```

---

## 3. Engine Portfolio

### 3.1 CopulaGenerator (Phase 1)

Port of the v1 engine with two improvements:
- **Gaussian copula option** via Spearman rank correlation → Pearson
  conversion, avoiding the need for complete-case filtering that `BetaCopula`
  requires.
- **Tables.jl input/output** instead of hard-coded `DataFrame`.

| Property | Value |
|----------|-------|
| Privacy | Public only |
| Sweet spot | N < 100k, D ≤ 30, rapid non-private benchmarking |
| Artifact | `FittedCopulaModel` |

### 3.2 MSTGenerator (Phase 2)

**Algorithm:** MST [McKenna et al. 2021].

1. Discretize continuous columns into `k`-bin histograms (default `k = 32`).
2. Select informative 2-way (or 3-way) marginals via the **exponential
   mechanism** (satisfies ε-DP).
3. Measure selected marginals with calibrated **Gaussian noise** (satisfies
   (ε,δ)-DP via zCDP composition [Bun & Steinke 2016]).
4. Construct a **junction tree** over selected marginal cliques.
5. Estimate the full joint distribution via belief propagation on the tree.
6. Sample synthetic rows from the reconstructed distribution and
   un-discretize continuous columns.

| Property | Value |
|----------|-------|
| Privacy | (ε,δ)-DP with tight zCDP composition |
| Sweet spot | ε ≤ 1.0, categorical-heavy or binned continuous, N < 50k |
| Artifact | `FittedMSTModel` |

### 3.3 DPCopulaGenerator (Phase 2)

1. Compute DP-noisy quantiles for each marginal via the smooth-sensitivity
   quantile mechanism [Smith 2011].
2. Compute a **private covariance matrix** via the Analyze-Gauss mechanism
   [Dwork et al. 2014].
3. Fit a Gaussian copula from the private covariance.

| Property | Value |
|----------|-------|
| Privacy | (ε,δ)-DP |
| Sweet spot | ε ≈ 2–4, continuous-heavy, N < 50k |
| Artifact | `FittedDPCopulaModel` |

### 3.4 DiffusionGenerator (Phase 3 — Extension)

TabDDPM architecture [Kotelnikov et al. 2023]. Gaussian diffusion for
numerical features [Ho et al. 2020], multinomial diffusion for categoricals
[Hoogeboom et al. 2021]. DP-SGD training via [Abadi et al. 2016] with
Rényi DP accounting [Mironov 2017]. Loaded only when `Lux.jl` is present.

| Property | Value |
|----------|-------|
| Privacy | Public or (ε,δ)-DP via DP-SGD |
| Sweet spot | D > 50, N > 50k, complex non-linear structure |
| Artifact | `FittedDiffusionModel` |

---

## 4. Public API

### 4.1 `fit`

```julia
function DataMimic.fit(
    generator::AbstractGenerator,
    table;
    privacy::Union{Nothing, PrivacyBudget} = nothing,
    hints::Vector{ColumnHint}              = ColumnHint[],
    identifiers::Vector{Symbol}            = Symbol[],
    fill::Dict{Symbol, FillSpec}            = Dict{Symbol, FillSpec}(),
    rng::AbstractRNG                       = Random.default_rng(),
) -> AbstractFittedModel
```

**Behavior:**

1. Validate `Tables.istable(table)` or throw.
2. Materialize column iterators via `Tables.columns(table)`.
3. Determine identifier columns — the union of:
   - Columns named in `identifiers`.
   - Columns tagged `ColumnHint(kind=:identifier)` in `hints`.
   - **Auto-detected:** string or integer columns where the number of
     distinct non-missing values is ≥ 90% of `N_nonmissing` for that
     column (high-cardinality heuristic). A `@info` message is logged
     when auto-detection fires so the user knows which columns were
     excluded.
4. For each **non-identifier** column: detect type (or use `hints`), profile
   missingness, fit marginal.
5. **Privacy / generator compatibility check:**
   - `AbstractPublicGenerator` + `privacy !== nothing` → error.
   - `AbstractPrivateGenerator` + `privacy === nothing` → error.
   - `DiffusionGenerator(dp=true)` + `privacy === nothing` → error.
   - `DiffusionGenerator(dp=false)` + `privacy !== nothing` → error.
6. If `AutoGenerator`, resolve to a concrete generator (§5).
7. Fit the engine-specific model and return a concrete `AbstractFittedModel`.

**Name-conflict note:** `DataMimic.fit` is a new function owned by this
package — it is *not* `StatsBase.fit`. If a user has both loaded, they
qualify: `DataMimic.fit(...)` or `StatsBase.fit(...)`. We do **not** extend
`StatsBase.fit` because our signature (`generator, table; ...`) does not
follow the StatsBase convention (`Type, data`).

### 4.2 `sample`

```julia
function DataMimic.sample(
    model::AbstractFittedModel,
    n::Int;
    rng::AbstractRNG = model.rng,
) -> table
```

1. Generate `n` synthetic rows from the fitted model.
2. Re-inject missing values at profiled rates.
3. Fill identifier columns according to their fill spec (§4.3).
4. Materialize output via `Tables.materializer` of the original input, falling
   back to `NamedTuple` of vectors when no materializer is available.

### 4.3 Identifier Column Handling

Identifier columns (SSNs, names, emails, account numbers, etc.) carry no
statistical signal. DataMimic **excludes them from the statistical model**
entirely — they are never fed to the copula, marginal fitter, or any
engine.

**In the output**, identifier columns are handled by the `fill` kwarg
on `fit()` (see §4.1 for the full signature). The `fill` dict maps
identifier column names to replacement strategies:

| Fill value | Behavior | Example output |
|------------|----------|----------------|
| *(not in `fill` dict)* | Column is **dropped** from output | — |
| `:sequential` | `"<colname>_1"`, `"<colname>_2"`, ... | `"id_1"`, `"id_2"` |
| `:sequential_int` | `1`, `2`, `3`, ... | `1`, `2`, `3` |
| `"prefix"` (any String) | `"prefix_1"`, `"prefix_2"`, ... | `"patient_1"`, `"patient_2"` |
| `f::Function` | `f(i)` called for row `i = 1:n` | `i -> "USER_$(lpad(i, 5, '0'))"` |

**Examples:**

```julia
# Drop identifiers entirely (default — safest)
model = fit(CopulaGenerator(), df; identifiers=[:ssn, :name])
sample(model, 100)  # output has no :ssn or :name columns

# Keep columns with sequential placeholders
model = fit(CopulaGenerator(), df;
    identifiers = [:ssn, :name],
    fill = Dict(:ssn => :sequential_int, :name => "person"),
)
sample(model, 100)  # :ssn = [1, 2, ...], :name = ["person_1", "person_2", ...]

# Custom generator function
model = fit(CopulaGenerator(), df;
    identifiers = [:patient_id],
    fill = Dict(:patient_id => i -> "SYNTH-$(lpad(i, 6, '0'))"),
)
sample(model, 100)  # :patient_id = ["SYNTH-000001", "SYNTH-000002", ...]
```

**Why not scramble?** v1's `scramble` shuffled characters/digits of real
values, producing output that was neither realistic (garbage strings) nor
truly private (preserved character frequencies, a side-channel). Excluding
identifiers from the model and filling with explicit placeholders is both
safer and more useful for downstream testing.

### 4.4 `synthesize` (convenience)

```julia
synthesize(generator, table, n; kw...) = sample(fit(generator, table; kw...), n)
```

### 4.5 Serialization

```julia
DataMimic.save(path::AbstractString, model::AbstractFittedModel)
DataMimic.load(path::AbstractString) -> AbstractFittedModel
```

Uses `Serialization.serialize` / `deserialize` with a version header so
older models can be detected and rejected with a clear message rather than
a corrupt-data crash. A future version may migrate to JLD2 if the
`Serialization` format proves too fragile across Julia versions.

**Note:** When a `fill` spec contains a `Function`, it is stored in the
model and serialized. Anonymous functions serialize correctly within the
same Julia version but may fail across versions — document this limitation.

---

## 5. AutoGenerator Dispatch

`AutoGenerator` inspects `(N, D, column_kinds, privacy)` and selects.
`D` counts only non-identifier columns.

| Privacy | Condition | Dispatched To | Rationale |
|---------|-----------|---------------|-----------|
| `nothing` | D ≤ 30 | `CopulaGenerator(:beta)` | Fast, deterministic, no neural overhead |
| `nothing` | D > 30 or N > 100k | `DiffusionGenerator(dp=false)` | Captures deep non-linear structure |
| `PrivacyBudget` | N < 20k, categorical fraction > 50% | `MSTGenerator(2)` | DP-SGD degrades at small N; marginal histograms preserve utility |
| `PrivacyBudget` | N < 20k, categorical fraction ≤ 50% | `DPCopulaGenerator()` | Avoids deep-learning noise penalty on small continuous tables |
| `PrivacyBudget` | N ≥ 20k, D > 30 | `DiffusionGenerator(dp=true)` | Sufficient density for DP-SGD convergence |
| `PrivacyBudget` | N ≥ 20k, D ≤ 30 | `MSTGenerator(2)` | Ample data but low dimension — PGM is cheaper and competitive |

When the selected engine lives in an unloaded extension (e.g., `DiffusionGenerator`
without Lux), `fit` throws with a message telling the user which package to load:

```
DiffusionGenerator requires Lux.jl. Run `using Lux` before calling fit.
```

**Phase-gating:** During Phase 1, only `CopulaGenerator` is available.
`AutoGenerator` enforces this:

- With a `PrivacyBudget`:
  ```
  Private generators are not yet implemented. They arrive in v2.0-beta (Phase 2).
  ```
- When dispatch would select `DiffusionGenerator` (the `D > 30` or
  `N > 100k` non-private path):
  ```
  DiffusionGenerator is not yet implemented. Use CopulaGenerator() directly,
  or wait for v2.0-rc (Phase 3).
  ```
  In Phase 1, `AutoGenerator` with no privacy always resolves to
  `CopulaGenerator(:beta)` regardless of `D` or `N`.

---

## 6. Evaluation Suite — `DataMimic.Evaluate` (Phase 4)

A submodule providing six standard metrics. All use lightweight
dependencies already in the core dep tree or `DecisionTree.jl` (6 deps,
all stdlib-level). No heavy ML framework required.

**Why not MLJ.jl?** MLJ pulls in 24 direct dependencies (MLJBase,
MLJModels, MLJTuning, MLJEnsembles, MLJTransforms, ScientificTypes,
OpenML, ...) — most of which are irrelevant for TSTR evaluation.
`DecisionTree.jl` provides random forests with a standalone API in 6
deps. Since TSTR compares "train on synth" vs. "train on real" using the
*same* model class, consistency matters more than peak model performance,
and a random forest is a standard TSTR baseline.

### 6.1 `fidelity_score(real, synth) -> NamedTuple`

- **1D:** Per-column Kolmogorov–Smirnov statistic (continuous) or Total
  Variation Distance (categorical).
- **2D:** Frobenius norm of the difference between real and synthetic
  pairwise Spearman correlation matrices.
- **Aggregate:** Weighted mean of 1D scores and the 2D score, returned
  alongside the per-column breakdown.
- **Deps:** `StatsBase`, `LinearAlgebra` (already in core).

### 6.2 `privacy_dcr(real, synth) -> NamedTuple`

- Computes the **Distance to Closest Record** (DCR) for every synthetic row.
- Returns the DCR vector, its median, 5th-percentile, and a count of
  exact matches (DCR = 0).
- **Deps:** `StatsBase`, `LinearAlgebra` (already in core).

### 6.3 `utility_tstr(real, synth, target; n_trees=100) -> NamedTuple`

- Trains a `RandomForestClassifier` or `RandomForestRegressor` from
  `DecisionTree.jl` on `synth`, evaluates on held-out `real`.
- Auto-detects classification vs. regression from the `target` column type.
- Returns accuracy (classification) or RMSE (regression) for both
  synth-trained and real-trained models, plus the ratio.
- Users who want TSTR with a different model can call `fidelity_score`
  and `privacy_dcr` for the statistical metrics and run their own
  train/evaluate loop with any ML framework.
- **Deps:** `DecisionTree.jl` (6 deps, all stdlib-level). Light enough
  to be a direct dependency rather than a package extension.

### 6.4 `jensen_shannon(real, synth) -> NamedTuple`

- Per-column **Jensen–Shannon divergence** for both continuous and
  categorical columns.
- Continuous columns are discretized into equal-width bins (default 50)
  before computing JSD; categorical columns use observed level
  frequencies directly.
- JSD is bounded [0, log(2)] and symmetrized — more interpretable than
  TVD for comparing distributions across datasets.
- Returns per-column JSD scores, the mean, and an aggregate.
- **Deps:** `StatsBase` (already in core).

### 6.5 `pairwise_marginal_error(real, synth; order=2) -> NamedTuple`

- Measures **2-way (or 3-way) contingency table error** between real
  and synthetic data.
- All columns are discretized (continuous → equal-width bins, categorical
  → levels), then every `order`-way combination of columns is compared
  via Total Variation Distance over the joint distribution.
- Returns per-pair TVD scores, the mean, and the worst-case pair — this
  directly validates MSTGenerator's core claim of preserving low-order
  marginal structure.
- **Deps:** `StatsBase` (already in core).

### 6.6 `privacy_utility_sweep(generator, table, epsilons, metric_fn; kw...) -> Vector{NamedTuple}`

- Runs a **privacy–utility curve**: fits and samples at each ε in
  `epsilons`, evaluating with `metric_fn` at each point.
- `metric_fn(real, synth)` can be any evaluation function
  (`fidelity_score`, `jensen_shannon`, etc.).
- Returns a vector of `(; epsilon, metric_result)` tuples — one per ε.
- Standard sweep values: `[0.1, 0.5, 1.0, 5.0, 10.0]`.
- **Deps:** None beyond what the chosen metric and generator require.

---

## 7. Project Structure

```
DataMimic/
├── Project.toml
├── src/
│   ├── DataMimic.jl          # module root, exports
│   ├── types.jl              # §2 type definitions
│   ├── detect.jl             # column-type detection (carried from v1)
│   ├── identifiers.jl        # identifier detection, fill specs
│   ├── privacy.jl            # PrivacyBudget, composition accounting
│   ├── fit.jl                # fit() dispatch + AutoGenerator routing
│   ├── sample.jl             # sample() dispatch
│   ├── serialize.jl          # save / load
│   ├── engines/
│   │   ├── copula.jl         # CopulaGenerator  → FittedCopulaModel
│   │   ├── mst.jl            # MSTGenerator     → FittedMSTModel      (Phase 2)
│   │   └── dp_copula.jl      # DPCopulaGenerator→ FittedDPCopulaModel (Phase 2)
│   └── evaluate/
│       ├── Evaluate.jl       # submodule root                         (Phase 4)
│       ├── fidelity.jl
│       ├── dcr.jl
│       └── tstr.jl           # TSTR via DecisionTree.jl
├── ext/
│   └── LuxExt.jl             # DiffusionGenerator engine              (Phase 3)
├── test/
│   ├── runtests.jl
│   ├── test_detect.jl
│   ├── test_copula.jl
│   ├── test_identifiers.jl   # identifier detection + fill specs
│   ├── test_tables.jl        # Tables.jl round-trip tests
│   ├── test_privacy.jl       # PrivacyBudget validation               (Phase 2)
│   ├── test_mst.jl           #                                        (Phase 2)
│   ├── test_diffusion.jl     #                                        (Phase 3)
│   └── test_evaluate.jl      #                                        (Phase 4)
└── PACKAGE_SPEC.md
```

---

## 8. Dependencies

### Core (always loaded)

| Package | Purpose |
|---------|---------|
| `Tables.jl` | Input/output abstraction |
| `Copulas.jl` | BetaCopula, GaussianCopula fitting |
| `StatsBase.jl` | countmap, Weights, sampling |
| `DataFrames.jl` | Direct dep — 95% of users will use it |
| `DecisionTree.jl` | Random forests for TSTR evaluation (6 stdlib-level deps) |
| `Random` | RNG threading |
| `Serialization` | Model save/load |
| `LinearAlgebra` | Covariance, PSD projection (DP copula) |

### Weak dependencies (package extensions)

| Package | Extension | Unlocks |
|---------|-----------|---------|
| `Lux.jl` | `LuxExt` | `DiffusionGenerator` |

---

## 9. Error Handling & Validation

| Condition | Behavior |
|-----------|----------|
| `!Tables.istable(input)` | `ArgumentError` |
| Zero rows or zero columns | `ArgumentError` |
| All columns are identifiers (zero statistical columns) | `ArgumentError("No statistical columns remain after excluding identifiers.")` |
| `identifiers` names a column not in the table | `ArgumentError` |
| `fill` key is not in resolved identifier set | `ArgumentError("fill key :foo is not an identifier column")` |
| `CopulaGenerator(copula_type)` with invalid symbol | `ArgumentError("copula_type must be :beta or :gaussian, got :foo")` |
| `MSTGenerator(order)` with `order ∉ {2, 3}` | `ArgumentError("max_marginal_order must be 2 or 3, got 5")` |
| `DiffusionGenerator(epochs=0)` or `batch_size=0` | `ArgumentError("epochs must be positive")` |
| `AbstractPublicGenerator` + `PrivacyBudget` | `ArgumentError("CopulaGenerator does not support privacy; use a private generator or remove the privacy budget.")` |
| `AbstractPrivateGenerator` + `privacy === nothing` | `ArgumentError("MSTGenerator requires a PrivacyBudget.")` |
| Extension engine not loaded | `ErrorException` with `using Lux` instructions |
| Phase 1 `AutoGenerator` + `PrivacyBudget` | `ErrorException("Private generators are not yet implemented.")` |
| Phase 1 `AutoGenerator` resolves to `DiffusionGenerator` | `ErrorException("DiffusionGenerator is not yet implemented. Use CopulaGenerator() directly.")` |
| `sample(model, n)` with `n < 1` | `ArgumentError` |
| `n > 10 × n_original` | `@warn` (empirical marginals will repeat) |
| Entirely-missing column | `@warn`, treated as constant(`missing`) |
| `ColumnHint` names a column not in the table | `ArgumentError` |
| `ColumnHint.levels` does not cover all observed values | `@warn` listing uncovered values; uncovered values excluded from marginal |
| Deserialized model version mismatch | `ErrorException` with upgrade instructions |
| Auto-detected identifier column | `@info("Column :foo auto-detected as identifier (N_unique/N_nonmissing = 0.98); excluding from model. Pass hints to override.")` |

---

## 10. Migration from v1.x

| v1.x | v2.0 |
|------|------|
| `fit(df; scramble=[:id])` | `fit(CopulaGenerator(), df; identifiers=[:id], fill=Dict(:id => :sequential))` |
| `fit(df)` | `fit(CopulaGenerator(), df)` |
| `sample(model, n)` | `sample(model, n)` (unchanged) |
| `synthesize(df, n)` | `synthesize(CopulaGenerator(), df, n)` or `synthesize(AutoGenerator(), df, n)` |
| Returns `DataFrame` always | Returns same type as input (DataFrame in, DataFrame out) |
| `SynthModel` | `FittedCopulaModel` (or other fitted type) |
| Global RNG | `rng=...` kwarg (global RNG is still the default) |
| `scramble` (char/digit shuffle) | Removed — use `identifiers` + `fill` instead (§4.3) |

---

## 11. References

Cited by section. Each entry is tagged with the engine or component that
depends on it so implementers know which papers to read for which phase.

### Copula-Based Synthesis (§3.1)

- **[Sklar 1959]** Sklar, A. "Fonctions de répartition à n dimensions et
  leurs marges." *Publications de l'Institut Statistique de l'Université
  de Paris*, 8, 229–231, 1959.
  — Foundational theorem: any joint distribution decomposes into marginals
  and a copula. Underpins `CopulaGenerator`. `Phase 1`

- **[Nelsen 2006]** Nelsen, R.B. *An Introduction to Copulas*. 2nd ed.,
  Springer, 2006.
  — Reference for Beta and Gaussian copula families used in
  `CopulaGenerator`. `Phase 1`

### MST / Graphical Model DP Synthesis (§3.2)

- **[McKenna et al. 2021]** McKenna, R., Miklau, G., Sheldon, D.
  "Winning the NIST Contest: A scalable and general approach to
  differentially private synthetic data." *Journal of Privacy and
  Confidentiality*, 11(3), 2021. Also presented at VLDB 2021.
  — The MST algorithm: exponential-mechanism marginal selection → Gaussian
  noise → junction tree → belief propagation. Primary reference for
  `MSTGenerator`. `Phase 2`

- **[Zhang et al. 2017]** Zhang, J., Cormode, G., Procopiuc, C.M.,
  Srivastava, D., Xiao, X. "PrivBayes: Private Data Release via Bayesian
  Networks." *ACM Transactions on Database Systems*, 42(4), 2017.
  — Predecessor to MST; useful for understanding the PGM-based synthesis
  approach. `Phase 2`

### DP Copula Synthesis (§3.3)

- **[Dwork et al. 2014]** Dwork, C., Talwar, K., Thakurta, A.,
  Zhang, L. "Analyze Gauss: Optimal Bounds for Privacy-Preserving
  Principal Component Analysis." *STOC 2014*, pp. 11–20.
  — Analyze-Gauss mechanism for private covariance estimation.
  Used by `DPCopulaGenerator`. `Phase 2`

- **[Smith 2011]** Smith, A. "Privacy-preserving statistical estimation
  with optimal convergence rates." *STOC 2011*, pp. 813–822.
  — Smooth-sensitivity quantile mechanism for DP-noisy marginals.
  Used by `DPCopulaGenerator`. `Phase 2`

### Diffusion Model Synthesis (§3.4)

- **[Kotelnikov et al. 2023]** Kotelnikov, A., Baranchuk, D.,
  Rubachev, I., Babenko, A. "TabDDPM: Modelling Tabular Data with
  Diffusion Models." *ICML 2023*.
  — TabDDPM architecture: Gaussian diffusion for numerical features,
  multinomial diffusion for categoricals, ResNet MLP backbone. Primary
  reference for `DiffusionGenerator`. `Phase 3`

- **[Ho et al. 2020]** Ho, J., Jain, A., Abbeel, P. "Denoising Diffusion
  Probabilistic Models." *NeurIPS 2020*.
  — Foundational DDPM paper: forward noising process, reverse denoising,
  linear β schedule. `Phase 3`

- **[Hoogeboom et al. 2021]** Hoogeboom, E., Nielsen, D., Jaini, P.,
  Forré, P., Welling, M. "Argmax Flows and Multinomial Diffusion."
  *NeurIPS 2021*.
  — Multinomial diffusion for categorical variables — the mechanism
  TabDDPM uses for non-numerical columns. `Phase 3`

### Differential Privacy Fundamentals

- **[Abadi et al. 2016]** Abadi, M., Chu, A., Goodfellow, I., McMahan,
  H.B., Mironov, I., Talwar, K., Zhang, L. "Deep Learning with
  Differential Privacy." *CCS 2016*, pp. 308–318.
  — DP-SGD algorithm: per-sample gradient clipping + Gaussian noise
  injection. Also introduces the moments accountant for tight ε
  composition. Used by `DiffusionGenerator(dp=true)`. `Phase 3`

- **[Mironov 2017]** Mironov, I. "Rényi Differential Privacy." *CSF
  2017*, pp. 263–275.
  — Rényi DP (zCDP) composition framework. Tighter privacy accounting
  than basic composition. Used by `MSTGenerator` and the DP-SGD
  accountant. `Phase 2–3`

- **[Bun & Steinke 2016]** Bun, M., Steinke, T. "Concentrated
  Differential Privacy: Simplifications, Extensions, and Lower Bounds."
  *TCC 2016*.
  — Zero-concentrated DP (zCDP). Basis for the composition accounting
  in `MSTGenerator`. `Phase 2`

### Evaluation Metrics (§6)

- **[Zhao et al. 2021]** Zhao, Z., Kunar, A., Birke, R., Chen, L.Y.
  "CTAB-GAN: Effective Table Data Synthesizing." *ACML 2021*.
  — Introduces the DCR (Distance to Closest Record) metric for
  measuring memorization in synthetic data. Used by `privacy_dcr`.
  `Phase 4`

- **[Esteban et al. 2017]** Esteban, C., Hyland, S.L., Rätsch, G.
  "Real-valued (Medical) Time Series Generation with Recurrent
  Conditional GANs." arXiv:1706.02633, 2017.
  — Formalizes the TSTR (Train on Synthetic, Test on Real) evaluation
  protocol. Used by `utility_tstr`. `Phase 4`

- **[Lin 1991]** Lin, J. "Divergence Measures Based on the Shannon
  Entropy." *IEEE Transactions on Information Theory*, 37(1), 1991.
  — Defines the Jensen–Shannon divergence. Used by `jensen_shannon`.
  `Phase 4`

- **[McKenna et al. 2019]** McKenna, R., Miklau, G., Sheldon, D.
  "Graphical-model based estimation and inference for differential
  privacy." *ICML 2019*.
  — Introduces pairwise marginal error as a key evaluation metric for
  DP synthesizers. Used by `pairwise_marginal_error`. `Phase 4`
