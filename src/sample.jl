using DataFrames
using StatsBase
using Random

# ─── Inverse-CDF helpers ──────────────────────────────────────────────────────

# Direct O(1) quantile from a pre-sorted vector using linear interpolation.
# This avoids the copy + partialsort! overhead that Statistics.quantile() incurs
# on every call when passed an unsorted (or unknown-sorted) vector.
@inline function _quantile_sorted(sv::Vector{Float64}, u::Float64)
    n    = length(sv)
    idx  = u * (n - 1)           # 0-based float position in sv
    lo   = floor(Int, idx) + 1   # 1-based lower index
    hi   = min(lo + 1, n)        # 1-based upper index (clamped at last element)
    frac = idx - (lo - 1)        # interpolation weight ∈ [0, 1)
    return sv[lo] + frac * (sv[hi] - sv[lo])
end

# Map a vector of uniform [0,1] values through the empirical quantile function.
function _invert_empirical(m::EmpiricalMarginal, u_vec::AbstractVector{Float64})
    sv = m.sorted_values
    return [_quantile_sorted(sv, clamp(u, 0.0, 1.0)) for u in u_vec]
end

# Convert raw Float64 quantile output to the column's original eltype.
function _cast_numeric(vals::Vector{Float64}, col_type::Symbol, T::Type)
    if col_type == :integer
        rounded = round.(Int64, vals)
        # Narrow back to original integer type if narrower than Int64.
        return T <: Integer ? convert(Vector{T}, rounded) :
                              convert(Vector{T}, Float64.(rounded))
    else
        return convert(Vector{T}, vals)
    end
end

# Draw n samples from a categorical marginal.
function _sample_categorical(m::CategoricalMarginal, n::Int)
    return StatsBase.sample(m.levels, Weights(m.probs), n)
end

# ─── Scramble helpers ─────────────────────────────────────────────────────────

# Scramble all characters of a string so the result is unrecognisable.
_scramble_value(v::AbstractString) = String(Random.shuffle(collect(v)))

# Scramble the decimal digits of an integer so the result is unrecognisable.
# Leading zeros produced by the scramble are dropped by parse(), which may
# reduce the magnitude (e.g. 10000 → "00001" → 1). This is intentional:
# the goal is to prevent the output from being a real identifier, not to
# preserve the exact range.
function _scramble_value(v::T) where T <: Integer
    digs     = collect(string(abs(v)))
    Random.shuffle!(digs)
    scrambled = parse(Int64, String(digs))
    return T(v < 0 ? -scrambled : scrambled)
end

# Fallback: other types (Float64, Bool, etc.) are passed through unchanged.
# fit() already warns when a scrambled column has one of these types.
_scramble_value(v) = v

# Apply scramble element-wise to a sampled column vector.
_apply_scramble(col::Vector) = [_scramble_value(v) for v in col]

# ─── Main sample function ─────────────────────────────────────────────────────

"""
    sample(model::SynthModel, nrows::Int) -> DataFrame

Draw `nrows` synthetic observations from a fitted `SynthModel`.
"""
function sample(model::SynthModel, nrows::Int)
    nrows < 1 && error("nrows must be at least 1.")

    if nrows > 10 * model.nrows_original
        @warn "Requested nrows ($nrows) is more than 10× the original " *
              "($(model.nrows_original) rows). Empirical marginals will repeat values."
    end

    col_names = model.column_names
    col_types = model.column_types
    result    = Dict{Symbol, Vector}()

    # Precompute name → index lookup to avoid O(ncols) findfirst per copula column.
    name_to_idx = Dict(n => i for (i, n) in enumerate(col_names))

    # ── Step 1–3: copula-based sampling of numeric columns ───────────────────
    if !isnothing(model.copula) && length(model.copula_columns) >= 2
        # rand returns (d × nrows) in column-major layout. Transposing to (nrows × d)
        # once means each per-column slice U_T[:, j] is contiguous in memory,
        # avoiding the cache-unfriendly row reads of U_mat[j, :].
        U_T = Matrix(rand(model.copula, nrows)')   # (nrows × d)

        for (j, cname) in enumerate(model.copula_columns)
            ctype = col_types[name_to_idx[cname]]
            m     = model.marginals[cname]::EmpiricalMarginal
            u_vec = U_T[:, j]                      # contiguous column slice
            vals  = _invert_empirical(m, u_vec)
            result[cname] = _cast_numeric(vals, ctype, m.original_eltype)
        end
    else
        # No copula: sample numeric columns independently.
        for (i, cname) in enumerate(col_names)
            ctype = col_types[i]
            ctype in (:continuous, :integer) || continue
            m     = model.marginals[cname]::EmpiricalMarginal
            u_vec = rand(nrows)
            vals  = _invert_empirical(m, u_vec)
            result[cname] = _cast_numeric(vals, ctype, m.original_eltype)
        end
    end

    # ── Step 4: sample categorical / binary / constant columns ───────────────
    for (i, cname) in enumerate(col_names)
        ctype = col_types[i]
        if ctype in (:categorical, :binary)
            m = model.marginals[cname]::CategoricalMarginal
            result[cname] = _sample_categorical(m, nrows)
        elseif ctype == :constant
            m = model.marginals[cname]::ConstantMarginal
            result[cname] = fill(m.value, nrows)
        end
    end

    # ── Step 4.5: scramble requested column values ────────────────────────────
    # Applied before missing re-injection so we only deal with concrete values.
    for cname in model.scrambled
        result[cname] = _apply_scramble(result[cname])
    end

    # ── Step 5: re-inject missings ────────────────────────────────────────────
    for cname in col_names
        p = model.missingness[cname]
        p > 0.0 || continue
        col      = result[cname]
        new_col  = allowmissing(col)
        mask     = rand(nrows) .< p
        new_col[mask] .= missing
        result[cname] = new_col
    end

    # ── Step 6: assemble DataFrame in original column order ───────────────────
    return DataFrame([cname => result[cname] for cname in col_names])
end

# ─── Convenience wrapper ──────────────────────────────────────────────────────

"""
    synthesize(df::DataFrame, nrows::Int; scramble=Symbol[]) -> DataFrame

Equivalent to `sample(fit(df; scramble=scramble), nrows)`.
"""
synthesize(df::DataFrame, nrows::Int; scramble::Vector{Symbol} = Symbol[]) =
    sample(fit(df; scramble=scramble), nrows)
