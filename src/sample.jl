using DataFrames
using StatsBase

# ─── Inverse-CDF helpers ──────────────────────────────────────────────────────

# Map a vector of uniform [0,1] values through the empirical quantile function.
function _invert_empirical(m::EmpiricalMarginal, u_vec::AbstractVector{Float64})
    # quantile() on a sorted vector uses linear interpolation.
    return [quantile(m.sorted_values, clamp(u, 0.0, 1.0)) for u in u_vec]
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

    # ── Step 1–3: copula-based sampling of numeric columns ───────────────────
    if !isnothing(model.copula) && length(model.copula_columns) >= 2
        # rand returns (d × nrows): each row is one variable, each column one obs.
        U_mat = rand(model.copula, nrows)   # (d, nrows)

        for (j, cname) in enumerate(model.copula_columns)
            idx    = findfirst(==(cname), col_names)
            ctype  = col_types[idx]
            m      = model.marginals[cname]::EmpiricalMarginal
            u_vec  = U_mat[j, :]            # nrows uniform values for this column
            vals   = _invert_empirical(m, u_vec)
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
    synthesize(df::DataFrame, nrows::Int) -> DataFrame

Equivalent to `sample(fit(df), nrows)`.
"""
synthesize(df::DataFrame, nrows::Int) = sample(fit(df), nrows)
