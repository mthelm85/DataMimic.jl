# DataMimic.jl v2.0 — Requirements

Derived from `PACKAGE_SPEC.md`. Every requirement uses EARS syntax and
carries a stable ID that never moves when rows are added or removed.
MoSCoW priority is scoped to **Phase 1 (v2.0-alpha)**: *Must* means the
alpha cannot ship without it; *Won't* means it is explicitly deferred to
a later phase.

**EARS patterns used:**

| Pattern | Template |
|---------|----------|
| Ubiquitous | The system shall \[action\]. |
| Event-driven | WHEN \[event\], the system shall \[action\]. |
| State-driven | WHILE \[state\], the system shall \[action\]. |
| Unwanted behavior | IF \[condition\], THEN the system shall \[action\]. |
| Optional feature | WHERE \[feature is included\], the system shall \[action\]. |

---

## 1. Tables.jl Integration

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-TBL-001** | `fit()` shall accept any input for which `Tables.istable` returns `true`. | Must | 1 |
| **REQ-TBL-002** | `fit()` shall materialize column iterators via `Tables.columns(table)`. | Must | 1 |
| **REQ-TBL-003** | `sample()` shall return the same concrete table type as the original input, resolved via `Tables.materializer`. | Must | 1 |
| **REQ-TBL-004** | IF `Tables.materializer` is not defined for the input type, THEN `sample()` shall fall back to a `NamedTuple` of vectors. | Must | 1 |
| **REQ-TBL-005** | IF the input does not satisfy `Tables.istable`, THEN `fit()` shall throw `ArgumentError`. | Must | 1 |

---

## 2. Type System

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-TYP-001** | The system shall define `AbstractGenerator` as the root type for all generator configurations. | Must | 1 |
| **REQ-TYP-002** | The system shall define `AbstractPublicGenerator <: AbstractGenerator` and `AbstractPrivateGenerator <: AbstractGenerator`. | Must | 1 |
| **REQ-TYP-003** | The system shall define `AbstractFittedModel` as the root type for all fitted model artifacts. | Must | 1 |
| **REQ-TYP-004** | Each concrete generator type shall produce its own concrete `AbstractFittedModel` subtype. | Must | 1 |
| **REQ-TYP-005** | `CopulaGenerator` shall accept a `copula_type::Symbol` parameter (`:beta` or `:gaussian`), defaulting to `:beta`. | Must | 1 |
| **REQ-TYP-006** | IF `CopulaGenerator` is constructed with a `copula_type` other than `:beta` or `:gaussian`, THEN the constructor shall throw `ArgumentError`. | Must | 1 |
| **REQ-TYP-007** | `AutoGenerator` shall subtype `AbstractGenerator` and carry no configuration fields. | Must | 1 |
| **REQ-TYP-008** | `MSTGenerator` shall subtype `AbstractPrivateGenerator` and accept `max_marginal_order::Int` (default `2`). | Done | 2 |
| **REQ-TYP-009** | `DPCopulaGenerator` shall subtype `AbstractPrivateGenerator` with no configuration fields. | Done | 2 |
| **REQ-TYP-010** | `DiffusionGenerator` shall subtype `AbstractGenerator` (not Public or Private) and accept `dp::Bool`, `epochs::Int`, `batch_size::Int`. | Done | 3 |
| **REQ-TYP-011** | `ColumnHint` shall accept `name::Symbol`, `kind::Symbol`, and optional `levels::Vector`. | Must | 1 |
| **REQ-TYP-012** | Valid `ColumnHint` `kind` values shall be `:continuous`, `:integer`, `:categorical`, `:binary`, `:constant`, `:identifier`. | Must | 1 |

---

## 3. Privacy Budget

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-PRV-001** | `PrivacyBudget` shall store `epsilon::Float64` and `delta::Float64`. | Must | 1 |
| **REQ-PRV-002** | `PrivacyBudget` shall default `delta` to `1e-5`. | Must | 1 |
| **REQ-PRV-003** | IF `epsilon ≤ 0`, THEN `PrivacyBudget` construction shall throw `ArgumentError`. | Must | 1 |
| **REQ-PRV-004** | IF `delta < 0` or `delta ≥ 1`, THEN `PrivacyBudget` construction shall throw `ArgumentError`. | Must | 1 |
| **REQ-PRV-005** | WHEN `fit()` is called with an `AbstractPublicGenerator` and `privacy !== nothing`, the system shall throw `ArgumentError` with message naming the generator and advising to use a private generator or remove the budget. | Should | 1 |
| **REQ-PRV-006** | WHEN `fit()` is called with an `AbstractPrivateGenerator` and `privacy === nothing`, the system shall throw `ArgumentError` naming the generator and stating it requires a `PrivacyBudget`. | Done | 2 |
| **REQ-PRV-007** | WHEN `fit()` is called with `DiffusionGenerator(dp=true)` and `privacy === nothing`, the system shall throw `ArgumentError`. | Done | 3 |
| **REQ-PRV-008** | WHEN `fit()` is called with `DiffusionGenerator(dp=false)` and `privacy !== nothing`, the system shall throw `ArgumentError`. | Done | 3 |

---

## 4. Column Detection

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-DET-001** | The system shall classify each non-identifier column into one of: `:continuous`, `:integer`, `:categorical`, `:binary`, `:constant`. | Must | 1 |
| **REQ-DET-002** | WHEN a column has zero non-missing values, the system shall classify it as `:constant`. | Must | 1 |
| **REQ-DET-003** | WHEN a column has exactly one unique non-missing value, the system shall classify it as `:constant`. | Must | 1 |
| **REQ-DET-004** | WHEN a column has exactly two unique non-missing values, the system shall classify it as `:binary`. | Must | 1 |
| **REQ-DET-005** | WHEN the non-missing base type is `<: AbstractFloat` and any value has a non-whole fractional component, the system shall classify the column as `:continuous`. | Must | 1 |
| **REQ-DET-006** | WHEN the non-missing base type is `<: AbstractFloat` and all values are whole numbers, the system shall classify the column as `:integer`. | Must | 1 |
| **REQ-DET-007** | WHEN the non-missing base type is `<: Integer`, the system shall classify the column as `:integer`. | Must | 1 |
| **REQ-DET-008** | WHEN the non-missing base type is `String`, `Symbol`, `Bool`, or `CategoricalValue` and the column has 3+ unique values, the system shall classify it as `:categorical`. | Must | 1 |
| **REQ-DET-009** | WHEN a `ColumnHint` is provided for a column with a `kind` other than `:identifier`, the hint `kind` shall override auto-detection for that column. | Must | 1 |

---

## 5. Identifier Handling

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-IDN-001** | The system shall exclude identifier columns from the statistical model entirely — they shall not be fed to the copula, marginal fitter, or any synthesis engine. | Must | 1 |
| **REQ-IDN-002** | The set of identifier columns shall be the union of: (a) columns named in the `identifiers` kwarg, (b) columns with `ColumnHint(kind=:identifier)`, and (c) auto-detected identifier columns. | Must | 1 |
| **REQ-IDN-003** | WHEN a string column has a number of distinct non-missing values ≥ 90% of `N`, `fit()` shall auto-detect it as an identifier. | Should | 1 |
| **REQ-IDN-004** | WHEN auto-detection classifies a column as an identifier, `fit()` shall emit `@info` naming the column and the distinct-value ratio. | Should | 1 |
| **REQ-IDN-005** | WHEN a `ColumnHint` with `kind` other than `:identifier` is provided for an auto-detected identifier column, the `ColumnHint` shall take precedence and the column shall be modeled normally. | Must | 1 |
| **REQ-IDN-006** | WHEN an identifier column has no entry in the `fill` dict, `sample()` shall drop that column from the output. | Must | 1 |
| **REQ-IDN-007** | WHEN `fill` maps an identifier column to `:sequential`, `sample()` shall fill it with `"<colname>_1"`, `"<colname>_2"`, …, `"<colname>_n"`. | Must | 1 |
| **REQ-IDN-008** | WHEN `fill` maps an identifier column to `:sequential_int`, `sample()` shall fill it with `1`, `2`, …, `n`. | Must | 1 |
| **REQ-IDN-009** | WHEN `fill` maps an identifier column to a `String` value, `sample()` shall use it as a prefix: `"<prefix>_1"`, `"<prefix>_2"`, …. | Must | 1 |
| **REQ-IDN-010** | WHEN `fill` maps an identifier column to a `Function`, `sample()` shall call `f(i)` for each row `i = 1:n`. | Must | 1 |
| **REQ-IDN-011** | IF `identifiers` names a column not present in the table, THEN `fit()` shall throw `ArgumentError`. | Must | 1 |
| **REQ-IDN-012** | IF a `fill` key does not appear in the resolved set of identifier columns, THEN `fit()` shall throw `ArgumentError`. | Must | 1 |
| **REQ-IDN-013** | `sample()` shall preserve the relative order of non-identifier columns from the original table. | Must | 1 |
| **REQ-IDN-014** | WHEN identifier columns are retained via `fill`, they shall appear in their original position in the column order. | Should | 1 |

---

## 6. Fitting — `fit()`

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-FIT-001** | `fit()` shall accept the positional arguments `(generator::AbstractGenerator, table)` and keyword arguments `privacy`, `hints`, `identifiers`, `fill`, `rng`. | Must | 1 |
| **REQ-FIT-002** | IF the input table has zero rows, THEN `fit()` shall throw `ArgumentError`. | Must | 1 |
| **REQ-FIT-003** | IF the input table has zero columns, THEN `fit()` shall throw `ArgumentError`. | Must | 1 |
| **REQ-FIT-004** | `fit()` shall profile the missingness rate (fraction of `missing` values) for each non-identifier column. | Must | 1 |
| **REQ-FIT-005** | `fit()` shall fit a marginal distribution for each non-identifier column based on its detected type. | Must | 1 |
| **REQ-FIT-006** | WHEN a column is entirely missing (`missingness = 1.0`), `fit()` shall emit `@warn` and treat it as `ConstantMarginal(missing)`. | Must | 1 |
| **REQ-FIT-007** | `fit()` shall return a concrete `AbstractFittedModel` subtype corresponding to the generator used. | Must | 1 |
| **REQ-FIT-008** | `DataMimic.fit` shall be a new function owned by the package — it shall not extend `StatsBase.fit`. | Must | 1 |
| **REQ-FIT-009** | IF a `ColumnHint` names a column not present in the table, THEN `fit()` shall throw `ArgumentError`. | Must | 1 |
| **REQ-FIT-010** | WHEN `AutoGenerator` is passed, `fit()` shall resolve it to a concrete generator before fitting (see §9). | Must | 1 |
| **REQ-FIT-011** | IF all non-identifier columns are removed (all columns are identifiers), THEN `fit()` shall throw `ArgumentError` stating no statistical columns remain. | Must | 1 |
| **REQ-FIT-012** | WHEN a `ColumnHint` provides `levels` for a `:categorical` column, the marginal shall use those levels and probabilities derived only from matching values. | Could | 1 |
| **REQ-FIT-013** | IF a `ColumnHint` provides `levels` that do not cover all observed values, THEN `fit()` shall emit `@warn` listing the uncovered values. | Could | 1 |

---

## 7. Sampling — `sample()`

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-SAM-001** | `sample()` shall accept `(model::AbstractFittedModel, n::Int; rng)` and return `n` synthetic rows. | Must | 1 |
| **REQ-SAM-002** | IF `n < 1`, THEN `sample()` shall throw `ArgumentError`. | Must | 1 |
| **REQ-SAM-003** | IF `n > 10 × model.n_original`, THEN `sample()` shall emit `@warn` that empirical marginals will repeat values. | Should | 1 |
| **REQ-SAM-004** | `sample()` shall re-inject `missing` values at the profiled rate for each column. | Must | 1 |
| **REQ-SAM-005** | `sample()` shall fill identifier columns according to their fill spec (REQ-IDN-006 through REQ-IDN-010) before materializing the output. | Must | 1 |
| **REQ-SAM-006** | `sample()` shall cast sampled numeric values back to the original column's non-missing eltype (e.g., `Float64` quantile output → `Int64` for `:integer` columns). | Must | 1 |
| **REQ-SAM-007** | `sample()` shall only produce categorical values drawn from the levels observed at `fit` time (or from `ColumnHint.levels` if supplied). | Must | 1 |
| **REQ-SAM-008** | WHILE a `:constant` column exists in the model, `sample()` shall fill it with the single observed value (including `missing` if that was the value). | Must | 1 |

---

## 8. Reproducibility / RNG

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-RNG-001** | Every stochastic public function (`fit`, `sample`, `synthesize`) shall accept an `rng::AbstractRNG` keyword argument. | Must | 1 |
| **REQ-RNG-002** | The default value of `rng` shall be `Random.default_rng()`. | Must | 1 |
| **REQ-RNG-003** | `sample()` shall default its `rng` to `model.rng` (the RNG stored at `fit` time). | Must | 1 |
| **REQ-RNG-004** | WHEN the same `rng` state is provided, `sample()` shall produce identical output. | Must | 1 |

---

## 9. AutoGenerator Dispatch

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-AUT-001** | `AutoGenerator` dispatch shall count only non-identifier columns when computing `D`. | Must | 1 |
| **REQ-AUT-002** | WHEN `privacy === nothing` and `D ≤ 30`, `AutoGenerator` shall resolve to `CopulaGenerator(:beta)`. | Must | 1 |
| **REQ-AUT-003** | WHEN `privacy === nothing` and (`D > 30` or `N > 100,000`), `AutoGenerator` shall resolve to `DiffusionGenerator(dp=false)`. | Done | 3 |
| **REQ-AUT-004** | WHEN `privacy !== nothing` and `N < 20,000` and categorical fraction > 50%, `AutoGenerator` shall resolve to `MSTGenerator(2)`. | Done | 2 |
| **REQ-AUT-005** | WHEN `privacy !== nothing` and `N < 20,000` and categorical fraction ≤ 50%, `AutoGenerator` shall resolve to `DPCopulaGenerator()`. | Done | 2 |
| **REQ-AUT-006** | WHEN `privacy !== nothing` and `N ≥ 20,000` and `D > 30`, `AutoGenerator` shall resolve to `DiffusionGenerator(dp=true)`. | Done | 3 |
| **REQ-AUT-007** | WHEN `privacy !== nothing` and `N ≥ 20,000` and `D ≤ 30`, `AutoGenerator` shall resolve to `MSTGenerator(2)`. | Done | 2 |
| **REQ-AUT-008** | WHILE Phase 1, IF `AutoGenerator` is called with a `PrivacyBudget`, THEN `fit()` shall throw `ErrorException` stating private generators arrive in v2.0-beta. | Must | 1 |
| **REQ-AUT-009** | WHILE Phase 1, IF `AutoGenerator` resolves to `DiffusionGenerator` (the non-private path), THEN `fit()` shall throw `ErrorException` stating DiffusionGenerator arrives in v2.0-rc and advising to use `CopulaGenerator` directly. | Must | 1 |
| **REQ-AUT-010** | IF the resolved engine requires an unloaded package extension, THEN `fit()` shall throw `ErrorException` naming the package to load (e.g., `"Run \`using Lux\` before calling fit."`). | Done | 3 |

---

## 10. Copula Engine

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-CPL-001** | `CopulaGenerator` shall fit independent empirical marginals (sorted non-missing values) for each numeric column. | Must | 1 |
| **REQ-CPL-002** | `CopulaGenerator(:beta)` shall fit a `BetaCopula` to model joint dependencies among numeric columns. | Must | 1 |
| **REQ-CPL-003** | `CopulaGenerator(:gaussian)` shall fit a Gaussian copula via Spearman rank correlation → Pearson conversion. | Should | 1 |
| **REQ-CPL-004** | IF fewer than 2 numeric (non-identifier) columns exist, THEN `CopulaGenerator` shall emit `@warn` and fall back to independent sampling. | Must | 1 |
| **REQ-CPL-005** | IF fewer than 2 complete cases exist across all numeric columns, THEN `CopulaGenerator` shall emit `@warn` and fall back to independent sampling. | Must | 1 |
| **REQ-CPL-006** | `CopulaGenerator` shall sample `:categorical` and `:binary` columns independently from their empirical probability distributions. | Must | 1 |
| **REQ-CPL-007** | `CopulaGenerator` shall sample `:constant` columns as `fill(value, n)`. | Must | 1 |
| **REQ-CPL-008** | WHEN sampling numeric columns with a copula, the system shall map uniform samples through each column's empirical quantile function (inverse CDF). | Must | 1 |
| **REQ-CPL-009** | `FittedCopulaModel` shall store: `column_names`, `column_kinds`, `marginals`, `missingness`, `copula`, `copula_columns`, `n_original`, `identifier_columns`, `identifier_fills`, `rng`. | Must | 1 |

---

## 11. MST Engine

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-MST-001** | `MSTGenerator` shall discretize continuous columns into `k`-bin histograms (default `k = 32`). | Done | 2 |
| **REQ-MST-002** | `MSTGenerator` shall select informative marginals via the exponential mechanism \[McKenna et al. 2021\]. | Done | 2 |
| **REQ-MST-003** | `MSTGenerator` shall measure selected marginals with calibrated Gaussian noise satisfying (ε,δ)-DP via zCDP composition \[Bun & Steinke 2016\]. | Done | 2 |
| **REQ-MST-004** | `MSTGenerator` shall construct a junction tree over selected marginal cliques. | Done | 2 |
| **REQ-MST-005** | `MSTGenerator` shall estimate the joint distribution via belief propagation on the junction tree. | Done | 2 |
| **REQ-MST-006** | `MSTGenerator` shall un-discretize continuous columns when sampling synthetic rows. | Done | 2 |
| **REQ-MST-007** | `MSTGenerator` shall support 2-way and 3-way marginals via the `max_marginal_order` parameter. | Done | 2 |

---

## 12. DP Copula Engine

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-DPC-001** | `DPCopulaGenerator` shall compute DP-noisy quantiles via the smooth-sensitivity mechanism \[Smith 2011\]. | Done | 2 |
| **REQ-DPC-002** | `DPCopulaGenerator` shall compute a private covariance matrix via the Analyze-Gauss mechanism \[Dwork et al. 2014\]. | Done | 2 |
| **REQ-DPC-003** | `DPCopulaGenerator` shall fit a Gaussian copula from the private covariance. | Done | 2 |

---

## 13. Diffusion Engine

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-DIF-001** | `DiffusionGenerator` shall implement the TabDDPM architecture \[Kotelnikov et al. 2023\]. | Done | 3 |
| **REQ-DIF-002** | `DiffusionGenerator` shall use Gaussian diffusion for numerical features \[Ho et al. 2020\]. | Done | 3 |
| **REQ-DIF-003** | `DiffusionGenerator` shall use multinomial diffusion for categorical features \[Hoogeboom et al. 2021\]. | Done | 3 |
| **REQ-DIF-004** | `DiffusionGenerator` shall use a ResNet-style MLP with sinusoidal timestep embedding as the denoising backbone. | Done | 3 |
| **REQ-DIF-005** | WHEN `dp=true`, `DiffusionGenerator` shall train using DP-SGD with per-sample gradient clipping and Gaussian noise injection \[Abadi et al. 2016\]. | Done | 3 |
| **REQ-DIF-006** | WHILE `dp=true`, `DiffusionGenerator` shall track cumulative privacy spend via Rényi DP accounting \[Mironov 2017\]. | Done | 3 |
| **REQ-DIF-007** | `DiffusionGenerator` shall be implemented as a Lux.jl package extension (`LuxExt`). | Done | 3 |
| **REQ-DIF-008** | IF `DiffusionGenerator` is requested and `Lux.jl` is not loaded, THEN `fit()` shall throw `ErrorException` with the message `"DiffusionGenerator requires Lux.jl. Run \`using Lux\` before calling fit."`. | Done | 3 |
| **REQ-DIF-009** | The `LuxExt` shall use `AutoZygote()` as the initial AD backend, with the architecture structured so switching to `AutoEnzyme()` is a single-token change. | Done | 3 |

---

## 14. Serialization

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-SER-001** | `DataMimic.save(path, model)` shall serialize an `AbstractFittedModel` to a file. | Should | 1 |
| **REQ-SER-002** | `DataMimic.load(path)` shall deserialize a file and return an `AbstractFittedModel`. | Should | 1 |
| **REQ-SER-003** | `save()` shall include a version header in the serialized output. | Should | 1 |
| **REQ-SER-004** | IF `load()` encounters a version mismatch, THEN it shall throw `ErrorException` with upgrade instructions rather than returning corrupt data. | Should | 1 |
| **REQ-SER-005** | WHEN a `fill` spec contains a `Function`, `save()` shall serialize it; the documentation shall note that anonymous functions may fail to deserialize across Julia versions. | Should | 1 |

---

## 15. Evaluation Suite

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-EVL-001** | `fidelity_score(real, synth)` shall compute per-column Kolmogorov–Smirnov statistics for continuous columns. | Done | 4 |
| **REQ-EVL-002** | `fidelity_score(real, synth)` shall compute per-column Total Variation Distance for categorical columns. | Done | 4 |
| **REQ-EVL-003** | `fidelity_score(real, synth)` shall compute the Frobenius norm of the difference between real and synthetic pairwise Spearman correlation matrices. | Done | 4 |
| **REQ-EVL-004** | `fidelity_score` shall return a `NamedTuple` containing per-column scores, a 2D correlation score, and a weighted aggregate. | Done | 4 |
| **REQ-EVL-005** | `privacy_dcr(real, synth)` shall compute the Distance to Closest Record for every synthetic row. | Done | 4 |
| **REQ-EVL-006** | `privacy_dcr` shall return a `NamedTuple` containing the DCR vector, its median, its 5th-percentile, and a count of exact matches (DCR = 0). | Done | 4 |
| **REQ-EVL-007** | `utility_tstr(real, synth, target)` shall train a `RandomForestClassifier` or `RandomForestRegressor` from `DecisionTree.jl` on `synth` and evaluate on held-out `real`. | Done | 4 |
| **REQ-EVL-008** | `utility_tstr` shall auto-detect classification vs. regression from the `target` column's element type. | Done | 4 |
| **REQ-EVL-009** | `utility_tstr` shall return a `NamedTuple` with accuracy or RMSE for both synth-trained and real-trained models, plus their ratio. | Done | 4 |
| **REQ-EVL-010** | `jensen_shannon(real, synth)` shall compute per-column Jensen–Shannon divergence, discretizing continuous columns into equal-width bins (default 50). | Must | 4b |
| **REQ-EVL-011** | `jensen_shannon` shall return a `NamedTuple` containing per-column JSD scores, the mean, and an aggregate. JSD values shall be bounded in \[0, log(2)\]. | Must | 4b |
| **REQ-EVL-012** | `pairwise_marginal_error(real, synth; order=2)` shall discretize all columns and compute Total Variation Distance over every `order`-way joint distribution. | Must | 4b |
| **REQ-EVL-013** | `pairwise_marginal_error` shall return a `NamedTuple` containing per-pair TVD scores, the mean, and the worst-case pair. | Must | 4b |
| **REQ-EVL-014** | `privacy_utility_sweep(generator, table, epsilons, metric_fn; kw...)` shall fit and sample at each ε, evaluate with `metric_fn`, and return a vector of `(; epsilon, metric_result)` tuples. | Must | 4b |
| **REQ-EVL-015** | `privacy_utility_sweep` shall accept any `metric_fn(real, synth)` that returns a `NamedTuple`, including `fidelity_score`, `jensen_shannon`, and user-defined functions. | Must | 4b |

---

## 16. Convenience API

| ID | Requirement | MoSCoW | Phase |
|----|-------------|--------|-------|
| **REQ-CON-001** | `synthesize(generator, table, n; kw...)` shall be equivalent to `sample(fit(generator, table; kw...), n)`. | Must | 1 |

---

## 17. Contradictions, Gaps, and Missing Edge Cases

Issues discovered during requirements extraction from `PACKAGE_SPEC.md`.

### Contradictions — all resolved in PACKAGE_SPEC.md

| ID | Issue | Resolution |
|----|-------|------------|
| ~~**GAP-001**~~ | **`fill` kwarg missing from §4.1 signature.** | ✅ Fixed — `fill::Dict{Symbol}` added to the §4.1 `fit()` signature; duplicate signature removed from §4.3. |
| ~~**GAP-002**~~ | **`PrivacyBudget` is Phase 2 but referenced in Phase 1 error paths.** | ✅ Fixed — `PrivacyBudget` struct moved to Phase 1 in the delivery table (shipped as a data type; no engines use it until Phase 2). REQ-PRV-001–004 are now Must/Phase 1. |
| ~~**GAP-003**~~ | **Design Principle 4 references MLJ.jl.** | ✅ Fixed — MLJ.jl removed from the principle; now reads "Heavy dependencies (Lux.jl)". |

### Missing Edge Cases — resolved in PACKAGE_SPEC.md

| ID | Issue | Resolution |
|----|-------|------------|
| ~~**GAP-004**~~ | **Integer identifier columns are not auto-detected.** | ✅ Fixed — auto-detection now covers string *and* integer columns. |
| ~~**GAP-005**~~ | **Missing values in identifier columns during auto-detection.** | ✅ Fixed — spec now reads "distinct non-missing values ≥ 90% of `N_nonmissing`". |
| ~~**GAP-006**~~ | **All columns are identifiers.** | ✅ Fixed — added to §9 error table: `ArgumentError("No statistical columns remain after excluding identifiers.")`. Covered by REQ-FIT-011. |
| ~~**GAP-007**~~ | **`copula_type` validation.** | ✅ Fixed — inner constructor added to `CopulaGenerator`; added to §9 error table. Covered by REQ-TYP-006. |
| ~~**GAP-008**~~ | **`DiffusionGenerator` parameter bounds.** | ✅ Fixed — inner constructor validates `epochs > 0`, `batch_size > 0`; added to §9 error table. |
| ~~**GAP-009**~~ | **`MSTGenerator` `max_marginal_order` bounds.** | ✅ Fixed — inner constructor validates `∈ {2, 3}`; added to §9 error table. |
| ~~**GAP-010**~~ | **`ColumnHint.levels` does not cover all observed values.** | ✅ Fixed — added to §9 error table: `@warn` listing uncovered values. Covered by REQ-FIT-013. |
| ~~**GAP-012**~~ | **Phase 1 `AutoGenerator` with `D > 30` and no privacy.** | ✅ Fixed — Phase 1 `AutoGenerator` always resolves to `CopulaGenerator(:beta)` regardless of `D`/`N`; added `ErrorException` for the `DiffusionGenerator` path. Covered by REQ-AUT-009. |

### Remaining open items (not contradictions — design decisions to document)

| ID | Issue | Suggested Resolution |
|----|-------|----------------------|
| **GAP-011** | **Identifier fill functions producing non-unique output.** A user-supplied `f(i)` could return duplicate values. | Do not enforce — fills are placeholders, not primary keys. Document that uniqueness is the caller's responsibility. |
| **GAP-013** | **Serialization of closures.** Closures capturing mutable external state could silently produce wrong output if the captured binding changes after `save()`. | Document: closures are serialized by value at `save()` time; mutations to captured variables after `save()` are not reflected on `load()`. |
| **GAP-014** | **`synthesize()` does not forward `rng`.** `rng` in `kw...` goes to `fit()` but `sample()` defaults to `model.rng`. | This is correct *if* `fit()` stores the provided `rng` in the model. Already implied by REQ-RNG-003 + REQ-CPL-009 — verify during implementation. |

---

## Summary Counts

| MoSCoW | Count |
|--------|-------|
| **Must** | 62 |
| **Should** | 9 |
| **Could** | 2 |
| **Done (previously Won't)** | 34 |
| **Total** | 107 |
| **Gaps found** | 14 (11 resolved, 3 open design decisions) |
