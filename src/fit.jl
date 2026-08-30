# ─── Privacy Validation ──────────────────────────────────────────────────────

function _validate_privacy(gen::AbstractPublicGenerator, privacy)
    if privacy !== nothing
        name = nameof(typeof(gen))
        throw(ArgumentError(
            "$name does not support privacy; " *
            "use a private generator or remove the privacy budget."))
    end
end

function _validate_privacy(gen::AbstractPrivateGenerator, privacy)
    if privacy === nothing
        name = nameof(typeof(gen))
        throw(ArgumentError("$name requires a PrivacyBudget."))
    end
end

function _validate_privacy(gen::DiffusionGenerator, privacy)
    if gen.dp && privacy === nothing
        throw(ArgumentError(
            "DiffusionGenerator with dp=true requires a PrivacyBudget."))
    end
    if !gen.dp && privacy !== nothing
        throw(ArgumentError(
            "DiffusionGenerator with dp=false does not support privacy; " *
            "set dp=true or remove the privacy budget."))
    end
end

function _validate_privacy(::AutoGenerator, privacy)
    # Handled in the AutoGenerator fit method
end

# ─── AutoGenerator ───────────────────────────────────────────────────────────

"""
    fit(::AutoGenerator, table; kw...)

Dispatch rules from PACKAGE_SPEC §5.

**Non-private**:
- D ≤ 30            →  `CopulaGenerator(:beta)`
- D > 30 or N > 100k →  `DiffusionGenerator(dp=false)`

**Private**:
- N < 20k, categorical fraction > 50%  →  `MSTGenerator(2)`
- N < 20k, categorical fraction ≤ 50%  →  `DPCopulaGenerator()`
- N ≥ 20k, D > 30                     →  `DiffusionGenerator(dp=true)`
- N ≥ 20k, D ≤ 30                     →  `MSTGenerator(2)`
"""
function fit(gen::AutoGenerator, table;
             privacy::Union{Nothing, PrivacyBudget} = nothing,
             hints::Vector{ColumnHint}              = ColumnHint[],
             identifiers::Vector{Symbol}            = Symbol[],
             fill                                   = Dict{Symbol, FillSpec}(),
             rng::AbstractRNG                       = Random.default_rng())

    # ── Inspect table for dispatch decision ──────────────────────────────
    Tables.istable(table) ||
        throw(ArgumentError("Input must be a Tables.jl-compatible table."))

    cols      = Tables.columns(table)
    col_names = collect(Symbol, Tables.columnnames(cols))
    N         = length(Tables.getcolumn(cols, first(col_names)))

    nm_cache       = Dict{Symbol, Vector}()
    basetype_cache = Dict{Symbol, Type}()
    for name in col_names
        col = Tables.getcolumn(cols, name)
        nm_cache[name]       = _nonmissing(col)
        basetype_cache[name] = _basetype(col)
    end

    id_set    = _resolve_identifiers(col_names, identifiers, hints,
                                      nm_cache, basetype_cache)
    stat_cols = filter(n -> !(n in id_set), col_names)
    D         = length(stat_cols)

    if privacy === nothing
        # ── Non-private dispatch ────────────────────────────────────────
        selected = if D > 30 || N > 100_000
            DiffusionGenerator(; dp = false)
        else
            CopulaGenerator(:beta)
        end
    else
        # ── Private dispatch ────────────────────────────────────────────
        hint_dict = Dict(h.name => h for h in hints)
        n_cat = 0
        for name in stat_cols
            h = get(hint_dict, name, nothing)
            kind = if h !== nothing && h.kind != :identifier
                h.kind
            else
                _detect_column_type(nm_cache[name], basetype_cache[name])
            end
            kind in (:categorical, :binary) && (n_cat += 1)
        end
        cat_frac = D > 0 ? n_cat / D : 0.0

        selected = if N < 20_000
            cat_frac > 0.5 ? MSTGenerator(2) : DPCopulaGenerator()
        else   # N ≥ 20k
            D > 30 ? DiffusionGenerator(; dp = true) : MSTGenerator(2)
        end
    end

    @info "AutoGenerator selected $(nameof(typeof(selected)))" columns=D rows=N private=(privacy !== nothing)

    return fit(selected, table;
               privacy = privacy, hints = hints,
               identifiers = identifiers, fill = fill, rng = rng)
end

# ─── Main fit ────────────────────────────────────────────────────────────────

"""
    fit(generator::AbstractGenerator, table; kw...) -> AbstractFittedModel

Fit a synthetic data model to `table` using the specified generator.

# Keywords
- `privacy::Union{Nothing, PrivacyBudget}=nothing` — required for private generators
- `hints::Vector{ColumnHint}=ColumnHint[]` — column type overrides
- `identifiers::Vector{Symbol}=Symbol[]` — columns to exclude from the model
- `fill=Dict{Symbol,FillSpec}()` — fill specs for identifier columns in output
- `rng::AbstractRNG=Random.default_rng()` — random number generator
"""
function fit(gen::AbstractGenerator, table;
             privacy::Union{Nothing, PrivacyBudget} = nothing,
             hints::Vector{ColumnHint}              = ColumnHint[],
             identifiers::Vector{Symbol}            = Symbol[],
             fill                                   = Dict{Symbol, FillSpec}(),
             rng::AbstractRNG                       = Random.default_rng())

    # ── 1. Validate input table ──────────────────────────────────────────
    Tables.istable(table) ||
        throw(ArgumentError("Input must be a Tables.jl-compatible table."))

    cols = Tables.columns(table)
    col_names = collect(Symbol, Tables.columnnames(cols))
    isempty(col_names) &&
        throw(ArgumentError("Input table has zero columns."))

    first_col = Tables.getcolumn(cols, first(col_names))
    nrows = length(first_col)
    nrows == 0 &&
        throw(ArgumentError("Input table has zero rows."))

    mat = Tables.materializer(table)

    # ── 2. Validate hint and identifier column names ─────────────────────
    for h in hints
        h.name in col_names ||
            throw(ArgumentError(
                "ColumnHint names column :$(h.name) which is not in the table."))
    end
    for id in identifiers
        id in col_names ||
            throw(ArgumentError(
                "identifiers names column :$id which is not in the table."))
    end

    # ── 3. Collect per-column metadata ───────────────────────────────────
    nm_cache       = Dict{Symbol, Vector}()
    basetype_cache = Dict{Symbol, Type}()

    for name in col_names
        col = Tables.getcolumn(cols, name)
        nm_cache[name]       = _nonmissing(col)
        basetype_cache[name] = _basetype(col)
    end

    # ── 4. Resolve identifiers ───────────────────────────────────────────
    id_set = _resolve_identifiers(col_names, identifiers, hints,
                                   nm_cache, basetype_cache)

    stat_cols = filter(n -> !(n in id_set), col_names)
    isempty(stat_cols) &&
        throw(ArgumentError(
            "No statistical columns remain after excluding identifiers."))

    # ── 5. Validate fill keys ────────────────────────────────────────────
    fill_dict = Dict{Symbol, FillSpec}()
    for (k, v) in pairs(fill)
        sk = Symbol(k)
        sk in id_set ||
            throw(ArgumentError("fill key :$sk is not an identifier column."))
        fill_dict[sk] = v
    end

    # ── 6. Validate privacy / generator compatibility ────────────────────
    _validate_privacy(gen, privacy)

    # ── 7. Dispatch to engine ────────────────────────────────────────────
    return _fit_engine(gen, cols, col_names, id_set, fill_dict,
                       hints, nm_cache, basetype_cache, nrows, mat, rng,
                       privacy)
end

# ─── CopulaGenerator engine integration ─────────────────────────────────────

function _fit_engine(gen::CopulaGenerator, cols, col_names, id_set, fill_dict,
                     hints, nm_cache, basetype_cache, nrows, mat, rng,
                     privacy)
    hint_dict    = Dict(h.name => h for h in hints)
    col_kinds    = Symbol[]
    marginals    = Dict{Symbol, Marginal}()
    miss         = Dict{Symbol, Float64}()
    copula_cols  = Symbol[]
    kind_of      = Dict{Symbol, Symbol}()

    for name in col_names
        if name in id_set
            push!(col_kinds, :identifier)
            continue
        end

        nm = nm_cache[name]
        T  = basetype_cache[name]
        n  = length(Tables.getcolumn(cols, name))
        p_miss = (n - length(nm)) / n
        miss[name] = p_miss

        if p_miss == 1.0
            @warn "Column :$name is entirely missing; treating as Constant(missing)."
        end

        # Determine kind — hint overrides auto-detection
        hint = get(hint_dict, name, nothing)
        if hint !== nothing && hint.kind != :identifier
            kind = hint.kind
            if hint.levels !== nothing
                observed  = unique(nm)
                uncovered = setdiff(observed, hint.levels)
                if !isempty(uncovered)
                    @warn "ColumnHint for :$name has levels that don't cover " *
                          "observed values: $uncovered; these values will be " *
                          "excluded from the marginal."
                end
            end
        else
            kind = _detect_column_type(nm, T)
        end
        push!(col_kinds, kind)

        # Categoricals join the copula via an ordinal encoding, so dependence
        # between them and the numeric columns is modelled rather than dropped.
        if kind in (:continuous, :integer, :categorical, :binary)
            push!(copula_cols, name)
            kind_of[name] = kind
        end

        marginals[name] = _fit_marginal(nm, kind, T; hint = hint)
    end

    # A categorical that collapsed to a single level carries no information for
    # the copula and would make its encoding degenerate.
    filter!(copula_cols) do name
        kind_of[name] in (:categorical, :binary) || return true
        length((marginals[name]::CategoricalMarginal).levels) >= 2
    end

    copula = _fit_copula(cols, copula_cols, kind_of, marginals,
                         nrows, gen.copula_type, rng)

    id_cols = [name for name in col_names if name in id_set]

    return FittedCopulaModel(
        col_names, col_kinds, marginals, miss,
        copula, copula_cols, nrows,
        id_cols, fill_dict, mat, rng,
    )
end

# ─── Fallback for DiffusionGenerator (extension not loaded) ──────────────────

function _fit_engine(::DiffusionGenerator, args...)
    error("DiffusionGenerator requires Lux.jl. " *
          "Run `using Lux, Zygote` before calling fit.")
end
