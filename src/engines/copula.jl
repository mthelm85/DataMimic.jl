# ─── CopulaGenerator Engine ─────────────────────────────────────────────────

# ─── Marginal fitting ───────────────────────────────────────────────────────

"""
Fit a marginal distribution for a single statistical column.

- `nm`   — pre-collected non-missing values
- `kind` — detected column kind (:continuous, :integer, :categorical, :binary, :constant)
- `T`    — non-missing eltype of the original column
- `hint` — optional ColumnHint (for locked levels)
"""
function _fit_marginal(nm::Vector, kind::Symbol, T::Type;
                       hint::Union{Nothing, ColumnHint} = nothing)
    if kind == :constant
        val = isempty(nm) ? missing : first(nm)
        return ConstantMarginal(val)

    elseif kind in (:continuous, :integer)
        sorted = sort!(Float64.(nm))
        return EmpiricalMarginal(sorted, T)

    else  # :categorical or :binary
        if hint !== nothing && hint.levels !== nothing
            level_set = Set(hint.levels)
            filtered = filter(v -> v in level_set, nm)
            if isempty(filtered)
                return ConstantMarginal(missing)
            end
            cm = StatsBase.countmap(filtered)
        else
            cm = StatsBase.countmap(nm)
        end
        lvls  = collect(keys(cm))
        probs = Float64.(collect(values(cm)))
        probs ./= sum(probs)
        return CategoricalMarginal(lvls, probs)
    end
end

# ─── Copula fitting ─────────────────────────────────────────────────────────

"""
Fit a copula to the numeric (non-identifier) columns.
Returns the fitted copula object, or `nothing` if fitting is not possible.
"""
function _fit_copula(cols, copula_columns::Vector{Symbol}, nrows::Int,
                     copula_type::Symbol)
    d = length(copula_columns)

    if d == 0
        @warn "No numeric columns found; all columns are categorical/constant. " *
              "Falling back to fully independent sampling."
        return nothing
    end

    if d == 1
        @warn "Only one numeric column present; copula fitting skipped."
        return nothing
    end

    # Build float matrix (nrows × d) with NaN for missing values
    X = Matrix{Float64}(undef, nrows, d)
    for (j, cname) in enumerate(copula_columns)
        col = Tables.getcolumn(cols, cname)
        for i in 1:nrows
            v = col[i]
            X[i, j] = ismissing(v) ? NaN : Float64(v)
        end
    end

    # Filter to complete cases (no NaN in any column)
    complete = vec(.!any(isnan.(X), dims = 2))
    Xc = X[complete, :]

    if size(Xc, 1) < 2
        @warn "Fewer than 2 complete cases for copula fitting; " *
              "using independent sampling."
        return nothing
    end

    # Copulas.jl convention: data matrix is (d × n_obs)
    U = Copulas.pseudos(Matrix(Xc'))

    if copula_type == :beta
        return StatsBase.fit(Copulas.BetaCopula, U)
    else  # :gaussian
        return StatsBase.fit(Copulas.GaussianCopula, U)
    end
end
