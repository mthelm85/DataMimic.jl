# DataMimic.jl v2.0 — Requirements

Derived from `PACKAGE_SPEC.md`. Every requirement uses EARS syntax and
carries a stable ID that never moves when rows are added or removed.

Each row carries two independent columns:

- **MoSCoW** — planning priority, scoped to **Phase 1 (v2.0-alpha)**. *Must*
  means the alpha cannot ship without it; *Won't* means it is explicitly
  deferred to a later phase. This does not change as work lands.
- **Status** — implementation state today: *Done*, *Partial*, *Not started*,
  or *Removed* for a requirement whose feature was deliberately withdrawn.

These were previously conflated in a single column, so a row reading `Must`
could mean either "high priority" or "not yet implemented" and there was no way
to tell which. Priorities are preserved as originally recorded; statuses were
verified against the source.

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

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-TBL-001** | `fit()` shall accept any input for which `Tables.istable` returns `true`. | Must | Done | 1 |
| **REQ-TBL-002** | `fit()` shall materialize column iterators via `Tables.columns(table)`. | Must | Done | 1 |
| **REQ-TBL-003** | `sample()` shall return the same concrete table type as the original input, resolved via `Tables.materializer`. | Must | Done | 1 |
| **REQ-TBL-004** | IF `Tables.materializer` is not defined for the input type, THEN `sample()` shall fall back to a `NamedTuple` of vectors. | Must | Done | 1 |
| **REQ-TBL-005** | IF the input does not satisfy `Tables.istable`, THEN `fit()` shall throw `ArgumentError`. | Must | Done | 1 |

---

## 2. Type System

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-TYP-001** | The system shall define `AbstractGenerator` as the root type for all generator configurations. | Must | Done | 1 |
| **REQ-TYP-002** | The system shall define `AbstractPublicGenerator <: AbstractGenerator` and `AbstractPrivateGenerator <: AbstractGenerator`. | Must | Done | 1 |
| **REQ-TYP-003** | The system shall define `AbstractFittedModel` as the root type for all fitted model artifacts. | Must | Done | 1 |
| **REQ-TYP-004** | Each concrete generator type shall produce its own concrete `AbstractFittedModel` subtype. | Must | Done | 1 |
| **REQ-TYP-005** | `CopulaGenerator` shall accept a `copula_type::Symbol` parameter (`:beta` or `:gaussian`), defaulting to `:beta`. | Must | Done | 1 |
| **REQ-TYP-006** | IF `CopulaGenerator` is constructed with a `copula_type` other than `:beta` or `:gaussian`, THEN the constructor shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-TYP-007** | `AutoGenerator` shall subtype `AbstractGenerator` and carry no configuration fields. | Must | Removed | 1 |
| **REQ-TYP-008** | `MSTGenerator` shall subtype `AbstractPrivateGenerator` and accept `max_marginal_order::Int` (default `2`). | Must | Done | 2 |
| **REQ-TYP-009** | `DPCopulaGenerator` shall subtype `AbstractPrivateGenerator` with no configuration fields. | Must | Done | 2 |
| **REQ-TYP-010** | `DiffusionGenerator` shall subtype `AbstractGenerator` (not Public or Private) and accept `dp::Bool`, `epochs::Int`, `batch_size::Int`, `hidden_dim::Int`, `n_blocks::Int`, `embed_dim::Int`, `dropout::Float64`, `lr::Float64`, `lr_warmup::Int`. | Must | Done | 3 |
| **REQ-TYP-011** | `ColumnHint` shall accept `name::Symbol`, `kind::Symbol`, and optional `levels::Vector`. | Must | Done | 1 |
| **REQ-TYP-012** | Valid `ColumnHint` `kind` values shall be `:continuous`, `:integer`, `:categorical`, `:binary`, `:constant`, `:identifier`. | Must | Done | 1 |

---

## 3. Privacy Budget

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-PRV-001** | `PrivacyBudget` shall store `epsilon::Float64` and `delta::Float64`. | Must | Done | 1 |
| **REQ-PRV-002** | `PrivacyBudget` shall default `delta` to `1e-5`. | Must | Done | 1 |
| **REQ-PRV-003** | IF `epsilon ≤ 0`, THEN `PrivacyBudget` construction shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-PRV-004** | IF `delta < 0` or `delta ≥ 1`, THEN `PrivacyBudget` construction shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-PRV-005** | WHEN `fit()` is called with an `AbstractPublicGenerator` and `privacy !== nothing`, the system shall throw `ArgumentError` with message naming the generator and advising to use a private generator or remove the budget. | Should | Done | 1 |
| **REQ-PRV-006** | WHEN `fit()` is called with an `AbstractPrivateGenerator` and `privacy === nothing`, the system shall throw `ArgumentError` naming the generator and stating it requires a `PrivacyBudget`. | Must | Done | 2 |
| **REQ-PRV-007** | WHEN `fit()` is called with `DiffusionGenerator(dp=true)` and `privacy === nothing`, the system shall throw `ArgumentError`. | Must | Done | 3 |
| **REQ-PRV-008** | WHEN `fit()` is called with `DiffusionGenerator(dp=false)` and `privacy !== nothing`, the system shall throw `ArgumentError`. | Must | Done | 3 |

---

## 4. Column Detection

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-DET-001** | The system shall classify each non-identifier column into one of: `:continuous`, `:integer`, `:categorical`, `:binary`, `:constant`. | Must | Done | 1 |
| **REQ-DET-002** | WHEN a column has zero non-missing values, the system shall classify it as `:constant`. | Must | Done | 1 |
| **REQ-DET-003** | WHEN a column has exactly one unique non-missing value, the system shall classify it as `:constant`. | Must | Done | 1 |
| **REQ-DET-004** | WHEN a column has exactly two unique non-missing values, the system shall classify it as `:binary`. | Must | Done | 1 |
| **REQ-DET-005** | WHEN the non-missing base type is `<: AbstractFloat` and any value has a non-whole fractional component, the system shall classify the column as `:continuous`. | Must | Done | 1 |
| **REQ-DET-006** | WHEN the non-missing base type is `<: AbstractFloat` and all values are whole numbers, the system shall classify the column as `:integer`. | Must | Done | 1 |
| **REQ-DET-007** | WHEN the non-missing base type is `<: Integer`, the system shall classify the column as `:integer`. | Must | Done | 1 |
| **REQ-DET-008** | WHEN the non-missing base type is `String`, `Symbol`, `Bool`, or `CategoricalValue` and the column has 3+ unique values, the system shall classify it as `:categorical`. | Must | Done | 1 |
| **REQ-DET-009** | WHEN a `ColumnHint` is provided for a column with a `kind` other than `:identifier`, the hint `kind` shall override auto-detection for that column. | Must | Done | 1 |

---

## 5. Identifier Handling

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-IDN-001** | The system shall exclude identifier columns from the statistical model entirely — they shall not be fed to the copula, marginal fitter, or any synthesis engine. | Must | Done | 1 |
| **REQ-IDN-002** | The set of identifier columns shall be the union of: (a) columns named in the `identifiers` kwarg, (b) columns with `ColumnHint(kind=:identifier)`, and (c) auto-detected identifier columns. | Must | Done | 1 |
| **REQ-IDN-003** | WHEN a string column has a number of distinct non-missing values ≥ 90% of `N`, `fit()` shall auto-detect it as an identifier. | Should | Done | 1 |
| **REQ-IDN-004** | WHEN auto-detection classifies a column as an identifier, `fit()` shall emit `@info` naming the column and the distinct-value ratio. | Should | Done | 1 |
| **REQ-IDN-005** | WHEN a `ColumnHint` with `kind` other than `:identifier` is provided for an auto-detected identifier column, the `ColumnHint` shall take precedence and the column shall be modeled normally. | Must | Done | 1 |
| **REQ-IDN-006** | WHEN an identifier column has no entry in the `fill` dict, `sample()` shall drop that column from the output. | Must | Done | 1 |
| **REQ-IDN-007** | WHEN `fill` maps an identifier column to `:sequential`, `sample()` shall fill it with `"<colname>_1"`, `"<colname>_2"`, …, `"<colname>_n"`. | Must | Done | 1 |
| **REQ-IDN-008** | WHEN `fill` maps an identifier column to `:sequential_int`, `sample()` shall fill it with `1`, `2`, …, `n`. | Must | Done | 1 |
| **REQ-IDN-009** | WHEN `fill` maps an identifier column to a `String` value, `sample()` shall use it as a prefix: `"<prefix>_1"`, `"<prefix>_2"`, …. | Must | Done | 1 |
| **REQ-IDN-010** | WHEN `fill` maps an identifier column to a `Function`, `sample()` shall call `f(i)` for each row `i = 1:n`. | Must | Done | 1 |
| **REQ-IDN-011** | IF `identifiers` names a column not present in the table, THEN `fit()` shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-IDN-012** | IF a `fill` key does not appear in the resolved set of identifier columns, THEN `fit()` shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-IDN-013** | `sample()` shall preserve the relative order of non-identifier columns from the original table. | Must | Done | 1 |
| **REQ-IDN-014** | WHEN identifier columns are retained via `fill`, they shall appear in their original position in the column order. | Should | Done | 1 |

---

## 6. Fitting — `fit()`

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-FIT-001** | `fit()` shall accept the positional arguments `(generator::AbstractGenerator, table)` and keyword arguments `privacy`, `hints`, `identifiers`, `fill`, `rng`. | Must | Done | 1 |
| **REQ-FIT-002** | IF the input table has zero rows, THEN `fit()` shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-FIT-003** | IF the input table has zero columns, THEN `fit()` shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-FIT-004** | `fit()` shall profile the missingness rate (fraction of `missing` values) for each non-identifier column. | Must | Done | 1 |
| **REQ-FIT-005** | `fit()` shall fit a marginal distribution for each non-identifier column based on its detected type. | Must | Done | 1 |
| **REQ-FIT-006** | WHEN a column is entirely missing (`missingness = 1.0`), `fit()` shall emit `@warn` and treat it as `ConstantMarginal(missing)`. | Must | Done | 1 |
| **REQ-FIT-007** | `fit()` shall return a concrete `AbstractFittedModel` subtype corresponding to the generator used. | Must | Done | 1 |
| **REQ-FIT-008** | `DataMimic.fit` shall be a new function owned by the package — it shall not extend `StatsBase.fit`. | Must | Done | 1 |
| **REQ-FIT-009** | IF a `ColumnHint` names a column not present in the table, THEN `fit()` shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-FIT-010** | WHEN `AutoGenerator` is passed, `fit()` shall resolve it to a concrete generator before fitting (see §9). | Must | Removed | 1 |
| **REQ-FIT-011** | IF all non-identifier columns are removed (all columns are identifiers), THEN `fit()` shall throw `ArgumentError` stating no statistical columns remain. | Must | Done | 1 |
| **REQ-FIT-012** | WHEN a `ColumnHint` provides `levels` for a `:categorical` column, the marginal shall use those levels and probabilities derived only from matching values. | Could | Done | 1 |
| **REQ-FIT-013** | IF a `ColumnHint` provides `levels` that do not cover all observed values, THEN `fit()` shall emit `@warn` listing the uncovered values. | Could | Done | 1 |

---

## 7. Sampling — `sample()`

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-SAM-001** | `sample()` shall accept `(model::AbstractFittedModel, n::Int; rng)` and return `n` synthetic rows. | Must | Done | 1 |
| **REQ-SAM-002** | IF `n < 1`, THEN `sample()` shall throw `ArgumentError`. | Must | Done | 1 |
| **REQ-SAM-003** | IF `n > 10 × model.n_original`, THEN `sample()` shall emit `@warn` that empirical marginals will repeat values. | Should | Done | 1 |
| **REQ-SAM-004** | `sample()` shall re-inject `missing` values at the profiled rate for each column. | Must | Done | 1 |
| **REQ-SAM-005** | `sample()` shall fill identifier columns according to their fill spec (REQ-IDN-006 through REQ-IDN-010) before materializing the output. | Must | Done | 1 |
| **REQ-SAM-006** | `sample()` shall cast sampled numeric values back to the original column's non-missing eltype (e.g., `Float64` quantile output → `Int64` for `:integer` columns). | Must | Done | 1 |
| **REQ-SAM-007** | `sample()` shall only produce categorical values drawn from the levels observed at `fit` time (or from `ColumnHint.levels` if supplied). | Must | Done | 1 |
| **REQ-SAM-008** | WHILE a `:constant` column exists in the model, `sample()` shall fill it with the single observed value (including `missing` if that was the value). | Must | Done | 1 |

---

## 8. Reproducibility / RNG

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-RNG-001** | Every stochastic public function (`fit`, `sample`, `synthesize`) shall accept an `rng::AbstractRNG` keyword argument. | Must | Done | 1 |
| **REQ-RNG-002** | The default value of `rng` shall be `Random.default_rng()`. | Must | Done | 1 |
| **REQ-RNG-003** | `sample()` shall default its `rng` to `model.rng` (the RNG stored at `fit` time). | Must | Done | 1 |
| **REQ-RNG-004** | WHEN the same `rng` state is provided, `sample()` shall produce identical output. | Must | Done | 1 |

---

## 9. Engine Selection *(withdrawn)*

`AutoGenerator` was removed before registration. Choosing an engine from table
shape alone was not defensible: measured rankings depend on the data, and for
private engines also on ε and row count, in ways `(N, D, column_kinds)` does
not predict. `compare` — which fits several engines to the user's own
table over repeated seeds — answers the same question with evidence instead of
a heuristic, so the heuristic was withdrawn rather than kept alongside it.

Rows below keep their IDs, per the stability rule above; only REQ-AUT-010,
which governs extension loading rather than dispatch, remains in force.

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-AUT-001** | `AutoGenerator` dispatch shall count only non-identifier columns when computing `D`. | Must | Removed | 1 |
| **REQ-AUT-002** | WHEN `privacy === nothing` and `D ≤ 30`, `AutoGenerator` shall resolve to `CopulaGenerator(:beta)`. | Must | Removed | 1 |
| **REQ-AUT-003** | WHEN `privacy === nothing` and (`D > 30` or `N > 100,000`), `AutoGenerator` shall resolve to `DiffusionGenerator(dp=false)`. | Must | Removed | 3 |
| **REQ-AUT-004** | WHEN `privacy !== nothing` and `N < 20,000` and categorical fraction > 50%, `AutoGenerator` shall resolve to `MSTGenerator(2)`. | Must | Removed | 2 |
| **REQ-AUT-005** | WHEN `privacy !== nothing` and `N < 20,000` and categorical fraction ≤ 50%, `AutoGenerator` shall resolve to `DPCopulaGenerator()`. | Must | Removed | 2 |
| **REQ-AUT-006** | WHEN `privacy !== nothing` and `N ≥ 20,000` and `D > 30`, `AutoGenerator` shall resolve to `DiffusionGenerator(dp=true)`. | Must | Removed | 3 |
| **REQ-AUT-007** | WHEN `privacy !== nothing` and `N ≥ 20,000` and `D ≤ 30`, `AutoGenerator` shall resolve to `MSTGenerator(2)`. | Must | Removed | 2 |
| **REQ-AUT-008** | WHILE Phase 1, IF `AutoGenerator` is called with a `PrivacyBudget`, THEN `fit()` shall throw `ErrorException` stating private generators arrive in v2.0-beta. | Must | Removed | 1 |
| **REQ-AUT-009** | WHILE Phase 1, IF `AutoGenerator` resolves to `DiffusionGenerator` (the non-private path), THEN `fit()` shall throw `ErrorException` stating DiffusionGenerator arrives in v2.0-rc and advising to use `CopulaGenerator` directly. | Must | Removed | 1 |
| **REQ-AUT-010** | IF the generator requires an unloaded package extension, THEN `fit()` shall throw `ErrorException` naming the package to load (e.g., `"Run \`using Lux\` before calling fit."`). | Must | Done | 3 |

---

## 10. Copula Engine

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-CPL-001** | `CopulaGenerator` shall fit independent empirical marginals (sorted non-missing values) for each numeric column. | Must | Done | 1 |
| **REQ-CPL-002** | `CopulaGenerator(:beta)` shall fit a `BetaCopula` to model joint dependencies among all modelled columns, numeric and categorical alike. | Must | Done | 1 |
| **REQ-CPL-003** | `CopulaGenerator(:gaussian)` shall fit a Gaussian copula to the pseudo-observations via `Copulas.jl`, which defaults to maximum likelihood on normal scores (`:mle`). Rank-inversion methods (`:irho` Spearman, `:itau` Kendall) are available in `Copulas.jl` but are not currently selected. | Should | Done | 1 |
| **REQ-CPL-004** | IF fewer than 2 modellable (non-identifier, non-constant) columns exist, THEN `CopulaGenerator` shall emit `@warn` and fall back to independent sampling. | Must | Done | 1 |
| **REQ-CPL-005** | IF fewer than 2 complete cases exist across all modelled columns, THEN `CopulaGenerator` shall emit `@warn` and fall back to independent sampling. | Must | Done | 1 |
| **REQ-CPL-006** | `CopulaGenerator` shall include `:categorical` and `:binary` columns in the copula via an ordinal encoding of their empirical CDF (the distributional transform), and shall sample them by inverting that CDF. A categorical column with fewer than 2 levels, or any categorical column when no copula could be fitted, shall instead be sampled independently from its empirical distribution. | Must | Done | 1 |
| **REQ-CPL-007** | `CopulaGenerator` shall sample `:constant` columns as `fill(value, n)`. | Must | Done | 1 |
| **REQ-CPL-008** | WHEN sampling with a copula, the system shall map uniform samples through each column's inverse CDF — the empirical quantile function for numeric columns, the level CDF for categorical ones. | Must | Done | 1 |
| **REQ-CPL-009** | `FittedCopulaModel` shall store: `column_names`, `column_kinds`, `marginals`, `missingness`, `copula`, `copula_columns`, `n_original`, `identifier_columns`, `identifier_fills`, `rng`. | Must | Done | 1 |

---

## 11. MST Engine

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-MST-001** | `MSTGenerator` shall discretize continuous columns into `k`-bin histograms (default `k = 32`). | Must | Done | 2 |
| **REQ-MST-002** | `MSTGenerator` shall select a spanning tree over columns via the exponential mechanism \[McKenna et al. 2021\]. | Must | Done | 2 |
| **REQ-MST-003** | `MSTGenerator` shall measure selected marginals with calibrated Gaussian noise satisfying (ε,δ)-DP via zCDP composition \[Bun & Steinke 2016\]. | Must | Done | 2 |
| **REQ-MST-004** | `MSTGenerator` shall construct a spanning tree over columns and store it as parent→child edges. | Must | Done | 2 |
| **REQ-MST-005** | `MSTGenerator` shall estimate the joint distribution by fitting a tree-structured Markov random field to all noisy measurements — every 1-way marginal and the selected 2-way marginals — via entropic mirror descent with exact sum-product belief propagation \[McKenna et al. 2019\], then sample ancestrally from the reconciled conditionals. Estimation is post-processing and consumes no privacy budget. | Must | Done | 2 |
| **REQ-MST-006** | `MSTGenerator` shall un-discretize continuous columns when sampling synthetic rows. | Must | Done | 2 |
| **REQ-MST-007** | `MSTGenerator` shall accept `max_marginal_order ∈ {2, 3}`; 3-way marginals are **not implemented** and fall back to 2-way with a warning. | Must | Partial | 2 |

> **Relationship to \[McKenna et al. 2021\].**  Checked against the reference
> implementation (`ryan112358/private-pgm`, `mechanisms/mst.py`).  All *d* 1-way
> marginals are measured, candidate edges are scored by count-scale L1 error
> against the independence reference, and the measurements are reconciled by
> Private-PGM before sampling.
>
> **Remaining difference: no domain compression.**  The reference merges bins
> whose noisy count falls below `3σ` into a single "other" category before
> selection, which matters on sparse categorical domains.  Not implemented here.
>
> The budget split is 30% selection / 20% 1-way / 50% 2-way, against the
> reference's ⅓ / ⅓ / ⅓.  The 1-way marginals serve mainly to anchor the
> selection score, so they take the smaller share; both are valid zCDP
> compositions.
>
> **Fixed — selection used to be effectively random.**  The exponential
> mechanism weights candidates by `exp(ε·q/(2Δ))`, so its ability to
> discriminate depends on the *absolute* spread of the score.  The original
> mutual-information score is measured in nats and spans a few tenths whatever
> the dataset size; with `ε_step = √(8·ρ_select/(d−1)) ≈ 0.03` on a 15-column
> table, every candidate landed within `exp(0.005)` of every other and the
> spanning tree was a uniform random draw.  Confirmed at the time by swapping
> the score function and getting **bit-identical** output at every ε.
>
> Scoring on the **count** scale — `‖M_ab(D) − ŷ_a ⊗ ŷ_b / n‖₁` against the
> noisy 1-way marginals, sensitivity 2 — makes the spread grow with `n`, which
> is what the reference implementation does.  The TSTR ratio now rises with the
> privacy budget (0.767 → 0.799 over ε ∈ [0.5, 8]) where it was previously flat
> at ≈0.79: extra budget had been buying nothing.  See `benchmark/eval_mst.jl`
> for before/after tables and for the seed-variance caveat at low ε.
>
> **PGM reconciliation now lands, because selection was fixed first.**  An
> earlier prototype of the same estimation code regressed TSTR by ≈0.10 and was
> rejected; re-tested on top of count-scale selection over 6 seeds per cell, the
> regression is gone entirely and it helps where noise dominates:
>
> | ε | fidelity off → on | TSTR off → on |
> |---|---|---|
> | 0.5 | 0.1513 → **0.1089** | 0.727 → **0.764** |
> | 1.0 | 0.1224 → **0.1077** | 0.771 → 0.785 |
> | 2.0 | 0.1124 → **0.1079** | 0.812 → 0.817 |
> | 4.0 | 0.1078 → 0.1077 | 0.814 → 0.814 |
> | 8.0 | **0.1057** → 0.1077 | 0.810 → 0.812 |
>
> The benefit scales inversely with the budget, as a variance-reduction step
> should: large at ε = 0.5 (fidelity 28% better, and the TSTR seed-standard-
> deviation halves from 0.064 to 0.029), negligible by ε = 4.  At ε = 8 fidelity
> is marginally *worse* (0.1057 vs 0.1077) — with near-exact measurements the
> binding constraint is tree-model misspecification rather than noise, so
> forcing the measurements onto the tree adds bias where raw conditionals were
> already fine.  Accepted: the regime that matters for a DP mechanism is the
> noisy one.

---

## 12. DP Copula Engine

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-DPC-001** | `DPCopulaGenerator` shall compute DP-noisy marginals via histogram binning with calibrated Gaussian noise (zCDP). | Must | Done | 2 |
| **REQ-DPC-002** | `DPCopulaGenerator` shall compute a private covariance matrix via the Analyze-Gauss mechanism \[Dwork et al. 2014\]. | Must | Done | 2 |
| **REQ-DPC-003** | `DPCopulaGenerator` shall fit a Gaussian copula from the private covariance. | Must | Done | 2 |

---

## 13. Diffusion Engine

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-DIF-001** | `DiffusionGenerator` shall implement the TabDDPM architecture \[Kotelnikov et al. 2023\]. | Must | Done | 3 |
| **REQ-DIF-002** | `DiffusionGenerator` shall use Gaussian diffusion for numerical features \[Ho et al. 2020\], with the ε parametrization, an MSE objective, and a cosine β schedule \[Nichol & Dhariwal 2021\]. | Must | Done | 3 |
| **REQ-DIF-003** | `DiffusionGenerator` shall use multinomial diffusion for categorical features \[Hoogeboom et al. 2021\], carrying categorical state in log space and training against the stochastic variational bound (`L_t / p_t + KL_prior`, normalized by the number of categorical features) under the `x0` parametrization. | Must | Done | 3 |
| **REQ-DIF-004** | `DiffusionGenerator` shall use the TabDDPM `MLPDiffusion` backbone: a plain MLP of `Dense → ReLU → Dropout` blocks (no normalization, no residual connections) with a sinusoidal timestep embedding added once at the input projection. | Must | Done | 3 |
| **REQ-DIF-005** | WHEN `dp=true`, `DiffusionGenerator` shall train using DP-SGD with per-sample gradient clipping and Gaussian noise injection \[Abadi et al. 2016\]. Lots shall be drawn by **Poisson subsampling** — each record included independently with probability `q = batch_size / n`, giving variable (possibly empty) lot sizes — so that the sampling mechanism matches the one the accountant models, and both the gradient average and the noise scale shall be normalized by the *expected* lot size `q · n` rather than the realized one. | Must | Done | 3 |
| **REQ-DIF-006** | WHILE `dp=true`, `DiffusionGenerator` shall track cumulative privacy spend via Rényi DP accounting for the Poisson-subsampled Gaussian mechanism \[Mironov 2017\], \[Mironov et al. 2019\]. The reported ε shall be a valid **upper bound** on the true spend: the closed-form RDP is exact at each integer order, but the order search is over a finite integer grid and the RDP → (ε, δ) conversion is the standard \[Mironov 2017, Prop. 3\] bound, both of which err conservatively. | Must | Done | 3 |
| **REQ-DIF-007** | `DiffusionGenerator` shall be implemented as a Lux.jl package extension (`LuxExt`). | Must | Done | 3 |
| **REQ-DIF-008** | IF `DiffusionGenerator` is requested and `Lux.jl` is not loaded, THEN `fit()` shall throw `ErrorException` with the message `"DiffusionGenerator requires Lux.jl. Run \`using Lux\` before calling fit."`. | Must | Done | 3 |
| **REQ-DIF-009** | The `LuxExt` shall use `AutoZygote()` as the initial AD backend, with the architecture structured so switching to `AutoEnzyme()` is a single-token change. | Must | Done | 3 |
| **REQ-DIF-010** | WHEN a GPU device is available (user has loaded `LuxCUDA`, `Metal.jl`, or `AMDGPU.jl`), the `LuxExt` shall auto-detect it via `Lux.gpu_device()` and move training data, model parameters, and optimizer state to the GPU. | Must | Done | 4b |
| **REQ-DIF-011** | WHEN training completes on GPU, the `LuxExt` shall move trained parameters back to CPU before storing them in `FittedDiffusionModel`. | Must | Done | 4b |
| **REQ-DIF-012** | WHEN sampling from a `FittedDiffusionModel`, the `LuxExt` shall move the model to the available device for the denoising loop, then move results back to CPU for post-processing. | Must | Done | 4b |
| **REQ-DIF-013** | GPU support shall not introduce any new dependencies on DataMimic — `LuxCUDA` is the user's opt-in, detected at runtime. | Must | Done | 4b |
| **REQ-DIF-014** | `DiffusionGenerator` shall anneal the learning rate linearly to zero (`lr · (1 - step/total)`), matching the reference trainer, with `lr_warmup` optionally prepending a linear warmup. | Must | Done | 4b |
| **REQ-DIF-015** | `DiffusionGenerator` shall expose network architecture hyperparameters (`d_layers`, `hidden_dim`, `n_blocks`, `embed_dim`, `dropout`, `num_timesteps`) for user tuning, with sensible defaults matching TabDDPM [Kotelnikov et al. 2023]. | Must | Done | 4b |
| **REQ-DIF-016** | `DiffusionGenerator` shall apply Gaussian quantile normalization to continuous features before training (empirical CDF → Φ⁻¹) and the inverse transform during sampling, matching TabDDPM §4.1 [Kotelnikov et al. 2023]. | Must | Done | 4b |
| **REQ-DIF-017** | WHEN `target` names a column, `DiffusionGenerator` shall train class-conditionally — adding `silu(label_emb(y))` to the timestep embedding — and sampling shall draw labels from the empirical class distribution before generating features conditioned on them. WHEN `target` is `nothing`, the model shall be unconditional. | Must | Done | 4c |
| **REQ-DIF-018** | `DiffusionGenerator` shall optimize with AdamW (`weight_decay`) and shall maintain an exponential moving average of the denoiser weights (`ema_decay`), using the EMA weights for sampling. | Must | Done | 4c |
| **REQ-DIF-019** | `sample(::FittedDiffusionModel, n)` shall run the full `num_timesteps` DDPM reverse process, using the Gaussian posterior mean/variance for numeric features and the multinomial posterior with Gumbel-max draws for categoricals. | Must | Done | 4c |

---

## 14. Serialization

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-SER-001** | `DataMimic.save(path, model)` shall serialize an `AbstractFittedModel` to a file. | Should | Done | 1 |
| **REQ-SER-002** | `DataMimic.load(path)` shall deserialize a file and return an `AbstractFittedModel`. | Should | Done | 1 |
| **REQ-SER-003** | `save()` shall include a version header in the serialized output. | Should | Done | 1 |
| **REQ-SER-004** | IF `load()` encounters a version mismatch, THEN it shall throw `ErrorException` with upgrade instructions rather than returning corrupt data. | Should | Done | 1 |
| **REQ-SER-005** | WHEN a `fill` spec contains a `Function`, `save()` shall serialize it; the documentation shall note that anonymous functions may fail to deserialize across Julia versions. | Should | Done | 1 |

---

## 15. Evaluation Suite

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-EVL-001** | `fidelity_score(real, synth)` shall compute per-column Kolmogorov–Smirnov statistics for continuous columns. | Must | Done | 4 |
| **REQ-EVL-002** | `fidelity_score(real, synth)` shall compute per-column Total Variation Distance for categorical columns. | Must | Done | 4 |
| **REQ-EVL-003** | `fidelity_score(real, synth)` shall compute the Frobenius norm of the difference between real and synthetic pairwise Spearman correlation matrices, excluding numeric columns with fewer than two distinct finite values in either table (their ranks are constant, so every correlation involving them is `0/0`). Excluded columns are still scored individually and are reported in `correlation_excluded`. | Must | Done | 4 |
| **REQ-EVL-004** | `fidelity_score` shall return a `NamedTuple` containing per-column scores, a 2D correlation score, and a weighted aggregate. | Must | Done | 4 |
| **REQ-EVL-005** | `privacy_dcr(real, synth)` shall compute the Distance to Closest Record for every synthetic row. | Must | Done | 4 |
| **REQ-EVL-006** | `privacy_dcr` shall return a `NamedTuple` containing the DCR vector, its median, its 5th-percentile, and a count of exact matches (DCR = 0). | Must | Done | 4 |
| **REQ-EVL-007** | `utility_tstr(real, synth, target)` shall train an `EvoTreeClassifier` or `EvoTreeRegressor` from `EvoTrees.jl` on `synth` and evaluate on held-out `real`, reporting macro-averaged F1 for classification (following TabDDPM [Kotelnikov et al. 2023]). Accepts an optional `test` kwarg for an external held-out test set; otherwise splits `real` 80/20 with stratified sampling. | Must | Done | 4 |
| **REQ-EVL-008** | `utility_tstr` shall auto-detect classification vs. regression from the `target` column's element type. | Must | Done | 4 |
| **REQ-EVL-009** | `utility_tstr` shall return a `NamedTuple` with accuracy or RMSE for both synth-trained and real-trained models, plus their ratio. | Must | Done | 4 |
| **REQ-EVL-010** | `jensen_shannon(real, synth)` shall compute per-column Jensen–Shannon divergence, discretizing continuous columns into equal-width bins (default 50). | Must | Done | 4b |
| **REQ-EVL-011** | `jensen_shannon` shall return a `NamedTuple` containing per-column JSD scores, the mean, and an aggregate. JSD values shall be bounded in \[0, log(2)\]. | Must | Done | 4b |
| **REQ-EVL-012** | `pairwise_marginal_error(real, synth; order=2)` shall discretize all columns and compute Total Variation Distance over every `order`-way joint distribution. | Must | Done | 4b |
| **REQ-EVL-013** | `pairwise_marginal_error` shall return a `NamedTuple` containing per-pair TVD scores, the mean, and the worst-case pair. | Must | Done | 4b |
| **REQ-EVL-014** | `privacy_utility_sweep(generator, table, epsilons, metric_fn; kw...)` shall fit and sample at each ε, evaluate with `metric_fn`, and return a vector of `(; epsilon, metric_result)` tuples. | Must | Done | 4b |
| **REQ-EVL-015** | `privacy_utility_sweep` shall accept any `metric_fn(real, synth)` that returns a `NamedTuple`, including `fidelity_score`, `jensen_shannon`, and user-defined functions. | Must | Done | 4b |

---

## 16. Convenience API

| ID | Requirement | MoSCoW | Status | Phase |
|----|-------------|--------|--------|-------|
| **REQ-CON-001** | `synthesize(generator, table, n; kw...)` shall be equivalent to `sample(fit(generator, table; kw...), n)`. | Must | Done | 1 |

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

Counted from the rows above.

| MoSCoW | Count |     | Status | Count |
|--------|-------|-----|--------|-------|
| **Must** | 129 |   | **Done** | 130 |
| **Should** | 11 |  | **Partial** | 1 |
| **Could** | 2 |    | **Removed** | 11 |
| **Total** | 142 |  | **Total** | 142 |

The single *Partial* is REQ-MST-007 (3-way marginals). The 11 *Removed* rows
are the withdrawn `AutoGenerator` dispatch (§9) plus its type and `fit`
requirements, REQ-TYP-007 and REQ-FIT-010.

**Gaps found**: 14 (11 resolved, 3 open design decisions).
