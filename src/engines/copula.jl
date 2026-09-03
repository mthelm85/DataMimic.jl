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
        sorted = sort!(_numeric.(nm))
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
Encode one column as pseudo-observations in `[0, 1]` for copula fitting.

Numeric columns use tied ranks, matching `Copulas.pseudos`.

Categorical and binary columns use the *distributional transform*: a level
occupying `[F(k-1), F(k)]` of the empirical CDF is mapped to a uniform draw
inside that interval. Mapping every observation of a level to a single point
instead would collapse the column to a handful of tied ranks and hide most of
its dependence from the copula.

The level order is taken from the column's fitted `CategoricalMarginal`, so
encoding here and inversion at sampling time agree by construction. The
association a copula can represent is monotone in that order, which is
arbitrary for a nominal variable — so this captures a real part of the
dependence, not all of it.
"""
function _encode_pseudo(vals::Vector{Float64}, kind::Symbol,
                        marginal, rng::AbstractRNG)
    n = length(vals)
    if kind in (:categorical, :binary)
        probs = marginal.probs
        cdf   = cumsum(probs)
        u = Vector{Float64}(undef, n)
        for i in 1:n
            k  = Int(vals[i])                 # 1-based level index
            lo = k == 1 ? 0.0 : cdf[k - 1]
            hi = cdf[k]
            u[i] = lo + rand(rng) * (hi - lo)
        end
        return u
    end
    return StatsBase.tiedrank(vals) ./ (n + 1)
end

"""
Fit a copula over the statistical (non-identifier) columns.

Categorical and binary columns participate through the ordinal encoding above,
so dependence between them and the numeric columns is modelled rather than
discarded. Returns the fitted copula, or `nothing` when fitting is not
possible.
"""
function _fit_copula(cols, copula_columns::Vector{Symbol},
                     kind_of::Dict{Symbol, Symbol},
                     marginals::Dict{Symbol, Marginal},
                     nrows::Int, copula_type::Symbol, rng::AbstractRNG)
    d = length(copula_columns)

    if d == 0
        @warn "No modellable columns found. " *
              "Falling back to fully independent sampling."
        return nothing
    end

    if d == 1
        @warn "Only one modellable column present; copula fitting skipped."
        return nothing
    end

    # Numeric values, or 1-based level indices for categoricals.  NaN marks a
    # value that cannot be placed (missing, non-finite, or an unseen level).
    X = Matrix{Float64}(undef, nrows, d)
    for (j, cname) in enumerate(copula_columns)
        col  = Tables.getcolumn(cols, cname)
        kind = kind_of[cname]
        if kind in (:categorical, :binary)
            m = marginals[cname]::CategoricalMarginal
            idx = Dict(v => k for (k, v) in enumerate(m.levels))
            for i in 1:nrows
                v = col[i]
                X[i, j] = ismissing(v) ? NaN : Float64(get(idx, v, NaN))
            end
        else
            for i in 1:nrows
                v = col[i]
                X[i, j] = ismissing(v) ? NaN : _numeric(v)
            end
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

    # Pseudo-observations, one column at a time so each kind is encoded
    # appropriately.  Copulas.jl expects a (d × n_obs) matrix.
    U = Matrix{Float64}(undef, d, size(Xc, 1))
    for (j, cname) in enumerate(copula_columns)
        U[j, :] = _encode_pseudo(Xc[:, j], kind_of[cname],
                                 get(marginals, cname, nothing), rng)
    end

    if copula_type == :beta
        return StatsBase.fit(Copulas.BetaCopula, U)
    end

    try
        return StatsBase.fit(Copulas.GaussianCopula, U)
    catch err
        err isa LinearAlgebra.PosDefException || rethrow()
    end

    # The Gaussian fit takes the correlation of the normal scores and factorizes
    # it. That correlation is singular whenever the copula columns are linearly
    # dependent after the rank transform, and the Cholesky then throws a bare
    # `PosDefException` from inside Distributions with nothing in it to act on.
    #
    # Two ordinary tables reach here: one with a duplicated or otherwise
    # collinear column, and one with fewer complete cases than columns. Neither
    # is a bad request, so repair the matrix rather than refusing. Found by a
    # sweep over OpenML tables, where four datasets of 13-15 rows crashed on
    # :gaussian while :beta handled all of them.
    R = Matrix{Float64}(StatsBase.cor(_normal_scores(U), dims = 2))

    # A constant column has zero variance, so its row and column come back NaN.
    # Independence is what a constant column carries, so write that in rather
    # than let the NaN reach the eigendecomposition.
    for i in axes(R, 1), j in axes(R, 2)
        isfinite(R[i, j]) || (R[i, j] = (i == j ? 1.0 : 0.0))
    end
    @warn "Gaussian copula: the $(d)x$(d) correlation matrix estimated from " *
          "$(size(Xc, 1)) complete case(s) is singular, so it was adjusted to " *
          "the nearest positive-definite correlation matrix. Collinear " *
          "columns, or fewer complete cases than columns, cause this. " *
          "Dependence among the affected columns is approximate; " *
          "CopulaGenerator(:beta) needs no such adjustment."
    return Copulas.GaussianCopula(_project_correlation(R))
end

# Standard normal quantile, via SpecialFunctions rather than a Distributions
# dependency the package does not otherwise carry. The clamp keeps a
# pseudo-observation that lands exactly on an endpoint - which the categorical
# encoder can produce - from becoming an infinite score and poisoning the whole
# correlation row.
_normal_scores(U) =
    sqrt(2.0) .* SpecialFunctions.erfinv.(2 .* clamp.(U, eps(), 1 - eps()) .- 1)
