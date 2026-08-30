# ─── Sampling Helpers ────────────────────────────────────────────────────────

# Direct O(1) quantile from a pre-sorted vector using linear interpolation.
@inline function _quantile_sorted(sv::Vector{Float64}, u::Float64)
    n    = length(sv)
    idx  = u * (n - 1)           # 0-based float position
    lo   = floor(Int, idx) + 1   # 1-based lower index
    hi   = min(lo + 1, n)        # clamped upper index
    frac = idx - (lo - 1)        # interpolation weight ∈ [0, 1)
    return sv[lo] + frac * (sv[hi] - sv[lo])
end

# Map uniform [0,1] values through the empirical quantile function.
function _invert_empirical(m::EmpiricalMarginal, u_vec::AbstractVector{Float64})
    sv = m.sorted_values
    return [_quantile_sorted(sv, clamp(u, 0.0, 1.0)) for u in u_vec]
end

# Cast Float64 quantile output to the column's original element type.
function _cast_numeric(vals::Vector{Float64}, col_kind::Symbol, T::Type)
    if col_kind == :integer
        rounded = round.(Int64, vals)
        return T <: Integer ? convert(Vector{T}, rounded) :
                              convert(Vector{T}, Float64.(rounded))
    else
        return convert(Vector{T}, vals)
    end
end

# Invert uniform draws through a CategoricalMarginal's CDF.
#
# The inverse of the encoding used for copula fitting: a draw landing in
# [F(k-1), F(k)] selects level k.  Level order comes from the same marginal, so
# this and _encode_pseudo agree by construction.
function _invert_categorical(m::CategoricalMarginal,
                             u_vec::AbstractVector{Float64})
    cdf  = cumsum(m.probs)
    nlev = length(m.levels)
    return [m.levels[clamp(searchsortedfirst(cdf, clamp(u, 0.0, 1.0)), 1, nlev)]
            for u in u_vec]
end

# Draw n samples from a categorical marginal.
function _sample_categorical(m::CategoricalMarginal, n::Int, rng::AbstractRNG)
    return StatsBase.sample(rng, m.levels, StatsBase.Weights(m.probs), n)
end

# ─── Common post-processing ─────────────────────────────────────────────────

"""
Re-inject missing values, fill identifiers, assemble columns in the
original order, and materialize.  Shared by all `sample` methods.
"""
function _postprocess(result::Dict{Symbol, Vector},
                      model, n::Int, rng::AbstractRNG)
    # ── Re-inject missing values ────────────────────────────────────────
    for (cname, p) in model.missingness
        p > 0.0 || continue
        haskey(result, cname) || continue
        col = result[cname]
        T   = eltype(col)
        new = Vector{Union{nonmissingtype(T), Missing}}(col)
        mask = rand(rng, n) .< p
        new[mask] .= missing
        result[cname] = new
    end

    # ── Fill identifier columns ─────────────────────────────────────────
    for cname in model.identifier_columns
        if haskey(model.identifier_fills, cname)
            result[cname] = _apply_fill(
                model.identifier_fills[cname], cname, n)
        end
    end

    # ── Assemble in original column order ───────────────────────────────
    output_names = Symbol[]
    output_cols  = Vector[]
    for cname in model.column_names
        if haskey(result, cname)
            push!(output_names, cname)
            push!(output_cols, result[cname])
        end
    end

    # ── Materialize to original table type ──────────────────────────────
    result_nt = NamedTuple{Tuple(output_names)}(Tuple(output_cols))
    return model.materializer(result_nt)
end

# ─── sample(::FittedCopulaModel) ────────────────────────────────────────────

"""
    sample(model::FittedCopulaModel, n::Int; rng=model.rng) -> table

Generate `n` synthetic rows from a fitted copula model. Returns the
same table type as the original input (via `Tables.materializer`).
"""
function sample(model::FittedCopulaModel, n::Int;
                rng::AbstractRNG = model.rng)
    n ≥ 1 || throw(ArgumentError("n must be at least 1, got $n"))

    if n > 10 * model.n_original
        @warn "Requested n ($n) is more than 10× the original " *
              "($(model.n_original) rows). Empirical marginals will " *
              "repeat values."
    end

    col_names = model.column_names
    col_kinds = model.column_kinds
    result    = Dict{Symbol, Vector}()

    name_to_idx = Dict(nm => i for (i, nm) in enumerate(col_names))

    # ── 1. Copula-based sampling ─────────────────────────────────────────
    #
    # The copula spans numeric *and* categorical columns; each is inverted
    # through the marginal it was encoded against at fit time.
    stat_numeric = model.copula_columns

    if !isnothing(model.copula) && length(stat_numeric) >= 2
        # rand returns (d × n); transpose to (n × d) for contiguous column slices
        U_T = Matrix(rand(rng, model.copula, n)')

        for (j, cname) in enumerate(stat_numeric)
            kind  = col_kinds[name_to_idx[cname]]
            u_vec = U_T[:, j]
            if kind in (:categorical, :binary)
                m = model.marginals[cname]::CategoricalMarginal
                result[cname] = _invert_categorical(m, u_vec)
            else
                m    = model.marginals[cname]::EmpiricalMarginal
                vals = _invert_empirical(m, u_vec)
                result[cname] = _cast_numeric(vals, kind, m.original_eltype)
            end
        end
    else
        # No copula: sample numeric columns independently
        for (i, cname) in enumerate(col_names)
            kind = col_kinds[i]
            kind == :identifier && continue
            kind in (:continuous, :integer) || continue
            m     = model.marginals[cname]::EmpiricalMarginal
            u_vec = rand(rng, n)
            vals  = _invert_empirical(m, u_vec)
            result[cname] = _cast_numeric(vals, kind, m.original_eltype)
        end
    end

    # ── 2. Sample remaining categorical / binary / constant columns ──────
    #
    # Categoricals carried by the copula were filled above; only those left
    # out of it (single-level, or no copula at all) are drawn independently.
    for (i, cname) in enumerate(col_names)
        kind = col_kinds[i]
        kind == :identifier && continue
        haskey(result, cname) && continue
        if kind in (:categorical, :binary)
            m = model.marginals[cname]::CategoricalMarginal
            result[cname] = _sample_categorical(m, n, rng)
        elseif kind == :constant
            m = model.marginals[cname]::ConstantMarginal
            result[cname] = fill(m.value, n)
        end
    end

    return _postprocess(result, model, n, rng)
end

# ─── sample(::FittedMSTModel) ───────────────────────────────────────────────

"""
    sample(model::FittedMSTModel, n::Int; rng=model.rng) -> table

Generate `n` synthetic rows from a fitted MST model. Samples from the
tree-structured joint distribution and un-discretizes back to the
original domain.
"""
function sample(model::FittedMSTModel, n::Int;
                rng::AbstractRNG = model.rng)
    n ≥ 1 || throw(ArgumentError("n must be at least 1, got $n"))

    if n > 10 * model.n_original
        @warn "Requested n ($n) is more than 10× the original " *
              "($(model.n_original) rows)."
    end

    stat_cols = model.stat_columns
    d = length(stat_cols)

    # ── 1. Sample discrete bin indices via the tree ──────────────────────
    sampled_bins = Dict{Int, Vector{Int}}()

    # Root
    root = model.root
    sampled_bins[root] = StatsBase.sample(
        rng, 1:length(model.root_marginal),
        StatsBase.Weights(model.root_marginal), n)

    # Children — tree_edges are in breadth-first order from construction
    for (parent, child) in model.tree_edges
        cond = model.conditionals[(parent, child)]
        parent_vals = sampled_bins[parent]
        n_child_bins = size(cond, 2)
        child_vals = Vector{Int}(undef, n)
        for i in 1:n
            pv = parent_vals[i]
            child_vals[i] = StatsBase.sample(
                rng, 1:n_child_bins,
                StatsBase.Weights(@view cond[pv, :]))
        end
        sampled_bins[child] = child_vals
    end

    # ── 2. Undiscretize and build result ─────────────────────────────────
    result = Dict{Symbol, Vector}()
    for (idx, name) in enumerate(stat_cols)
        info = model.discretization[name]
        bins = sampled_bins[idx]
        result[name] = _undiscretize(bins, info, n, rng)
    end

    return _postprocess(result, model, n, rng)
end

# ─── sample(::FittedDPCopulaModel) ──────────────────────────────────────────

"""
    sample(model::FittedDPCopulaModel, n::Int; rng=model.rng) -> table

Generate `n` synthetic rows from a fitted DP-copula model. Uses the
private Gaussian copula for numeric columns and noisy categoricals.
"""
function sample(model::FittedDPCopulaModel, n::Int;
                rng::AbstractRNG = model.rng)
    n ≥ 1 || throw(ArgumentError("n must be at least 1, got $n"))

    if n > 10 * model.n_original
        @warn "Requested n ($n) is more than 10× the original " *
              "($(model.n_original) rows)."
    end

    col_names = model.column_names
    col_kinds = model.column_kinds
    result    = Dict{Symbol, Vector}()

    name_to_idx = Dict(nm => i for (i, nm) in enumerate(col_names))

    # ── 1. Copula-based numeric sampling ─────────────────────────────────
    stat_numeric = model.copula_columns

    if !isnothing(model.copula) && length(stat_numeric) >= 2
        U_T = Matrix(rand(rng, model.copula, n)')
        for (j, cname) in enumerate(stat_numeric)
            m = model.marginals[cname]::DPHistogramMarginal
            result[cname] = _invert_dp_marginal(m, U_T[:, j], rng)
        end
    else
        for (i, cname) in enumerate(col_names)
            kind = col_kinds[i]
            kind == :identifier && continue
            kind in (:continuous, :integer) || continue
            m = model.marginals[cname]
            if m isa DPHistogramMarginal
                result[cname] = _invert_dp_marginal(m, rand(rng, n), rng)
            elseif m isa ConstantMarginal
                result[cname] = fill(m.value, n)
            end
        end
    end

    # ── 2. Categorical / binary / constant ───────────────────────────────
    for (i, cname) in enumerate(col_names)
        kind = col_kinds[i]
        kind == :identifier && continue
        if kind in (:categorical, :binary)
            m = model.marginals[cname]::CategoricalMarginal
            result[cname] = _sample_categorical(m, n, rng)
        elseif kind == :constant
            m = model.marginals[cname]::ConstantMarginal
            result[cname] = fill(m.value, n)
        end
    end

    return _postprocess(result, model, n, rng)
end

# ─── Convenience ─────────────────────────────────────────────────────────────

"""
    synthesize(generator, table, n; kw...) -> table

Equivalent to `sample(fit(generator, table; kw...), n)`.
"""
function synthesize(generator::AbstractGenerator, table, n::Int; kw...)
    return sample(fit(generator, table; kw...), n)
end
