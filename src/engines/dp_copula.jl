# ─── DPCopulaGenerator Engine ──────────────────────────────────────────────
#
# DP-noisy histogram marginals + Analyze-Gauss private covariance
# → Gaussian copula.
#
# REQ-DPC-001 through REQ-DPC-003.

const DP_COPULA_BINS = 32

# ─── DP marginal fitting ──────────────────────────────────────────────────

"""
Fit a single column's marginal with DP noise.

Continuous/integer → `DPHistogramMarginal` (k-bin histogram + Gaussian noise).
Categorical/binary → `CategoricalMarginal` with noisy counts.
Constant → `ConstantMarginal`.
"""
function _fit_dp_marginal(nm::Vector, kind::Symbol, T::Type,
                          rho::Float64, rng::AbstractRNG;
                          hint::Union{Nothing, ColumnHint} = nothing)
    if kind == :constant
        val = isempty(nm) ? missing : first(nm)
        return ConstantMarginal(val)
    end

    if kind in (:continuous, :integer)
        k    = DP_COPULA_BINS
        vals = _numeric.(nm)
        lo, hi = extrema(vals)
        lo == hi && return ConstantMarginal(first(nm))

        edges = collect(range(lo, hi; length = k + 1))
        edges[1]   -= abs(lo) * 1e-10 + 1e-15
        edges[end] += abs(hi) * 1e-10 + 1e-15

        counts = zeros(k)
        for v in vals
            counts[_find_bin(v, edges)] += 1.0
        end

        sigma = _rho_to_sigma(rho, 1.0)
        counts .+= randn(rng, k) .* sigma
        counts .= max.(counts, 0.0)
        s = sum(counts)
        probs = s > 0 ? counts / s : fill(1.0 / k, k)

        return DPHistogramMarginal(edges, probs, T)
    end

    # ── Categorical / binary ────────────────────────────────────────────
    if hint !== nothing && hint.levels !== nothing
        level_set = Set(hint.levels)
        filtered  = filter(v -> v in level_set, nm)
        if isempty(filtered)
            return ConstantMarginal(missing)
        end
        cm = StatsBase.countmap(filtered)
        for lvl in hint.levels
            get!(cm, lvl, 0)
        end
    else
        cm = StatsBase.countmap(nm)
    end

    # As in `_fit_marginal`: an explicit `levels` hint carries an ordering,
    # so it wins over sorting.
    lvls = if hint !== nothing && hint.levels !== nothing
        [l for l in hint.levels if haskey(cm, l)]
    else
        _ordered_levels(cm)
    end
    counts = Float64[cm[l] for l in lvls]

    sigma = _rho_to_sigma(rho, 1.0)
    counts .+= randn(rng, length(counts)) .* sigma
    counts .= max.(counts, 0.0)
    s = sum(counts)
    probs = s > 0 ? counts / s : fill(1.0 / length(counts), length(counts))

    return CategoricalMarginal(lvls, probs)
end

# ─── DP CDF for rank-transforming data ────────────────────────────────────

"""
Evaluate the CDF implied by a `DPHistogramMarginal` at `val`.
Linear interpolation within each bin.
"""
function _dp_cdf(m::DPHistogramMarginal, val::Float64)
    edges = m.bin_edges
    probs = m.probs
    k = length(probs)
    cumprob = 0.0
    for i in 1:k
        if val < edges[i + 1]
            frac = (val - edges[i]) / (edges[i + 1] - edges[i])
            return clamp(cumprob + frac * probs[i], 0.0, 1.0)
        end
        cumprob += probs[i]
    end
    return 1.0
end

# ─── Inverse DP CDF for sampling ─────────────────────────────────────────

"""
Map uniform [0,1] samples through the inverse CDF of a
`DPHistogramMarginal`.  Each sample is mapped to a bin via the CDF,
then placed uniformly within that bin.
"""
function _invert_dp_marginal(m::DPHistogramMarginal,
                             u_vec::AbstractVector{Float64},
                             rng::AbstractRNG)
    edges = m.bin_edges
    probs = m.probs
    k = length(probs)
    cdf = cumsum(probs)
    n = length(u_vec)

    vals = Vector{Float64}(undef, n)
    for i in 1:n
        u = clamp(u_vec[i], 0.0, 1.0)
        b = searchsortedfirst(cdf, u)
        b = clamp(b, 1, k)
        lo, hi = edges[b], edges[b + 1]
        vals[i] = lo + rand(rng) * (hi - lo)
    end

    T = m.original_eltype
    _is_temporal(T) && return [_from_temporal(T, v) for v in vals]
    if T <: Integer
        return round.(T, vals)
    elseif T <: AbstractFloat
        return convert.(T, vals)
    else
        return vals
    end
end

# ─── Private covariance copula (Analyze-Gauss) ───────────────────────────

"""
Build a Gaussian copula from a private second-moment matrix (Analyze-Gauss,
[Dwork et al. 2014]).

1. Rank-transform data to [0,1] via the DP marginal CDF.  The marginals are
   already private, so this is post-processing and costs no budget.
2. Form the **uncentered** second-moment matrix `M = XᵀX / n`.
3. Add symmetric Gaussian noise calibrated to `rho_cov`.
4. Project to a valid correlation matrix and construct a `GaussianCopula`.

# Sensitivity

Analyze-Gauss calibrates to the Frobenius sensitivity of the released matrix.
Replacing one record `x` with `x′`, both in `[0,1]^d`, changes `M` by

    ΔM = (x xᵀ − x′ x′ᵀ) / n,   ‖ΔM‖_F ≤ (‖x‖² + ‖x′‖²) / n ≤ 2d / n

and for the add/remove-one neighbouring relation used here a single record
contributes `‖x xᵀ‖_F = ‖x‖² ≤ d`, giving `‖ΔM‖_F ≤ d / n`.  That is the bound
applied below.

The matrix is deliberately **not** mean-centered.  Centering by a sample mean
computed from the raw data would leak: the mean is data-dependent and released
implicitly through the centered matrix, and the `d / n` bound above does not
account for it.  Privatizing the mean separately would require splitting
`rho_cov`; releasing uncentered second moments avoids the question entirely and
is the form the cited mechanism analyses.  Since the columns are marginal CDF
values in `[0,1]`, `_project_correlation` rescales to unit diagonal afterwards.
"""
function _fit_dp_covariance_copula(cols, copula_columns::Vector{Symbol},
                                   marginals::Dict{Symbol, DPMarginal}, nrows::Int,
                                   rho_cov::Float64, rng::AbstractRNG)
    d = length(copula_columns)

    X = Matrix{Float64}(undef, nrows, d)
    complete = trues(nrows)
    for (j, cname) in enumerate(copula_columns)
        col = Tables.getcolumn(cols, cname)
        m   = marginals[cname]::DPHistogramMarginal
        for i in 1:nrows
            v = col[i]
            if ismissing(v) || (v isa AbstractFloat && !isfinite(v))
                complete[i] = false
                X[i, j] = NaN
            else
                X[i, j] = _dp_cdf(m, _numeric(v))
            end
        end
    end

    Xc = X[complete, :]
    nc = size(Xc, 1)
    if nc < d + 1
        @warn "Fewer than d+1 complete cases for DP copula; " *
              "using independent sampling."
        return nothing
    end

    # Uncentered second-moment matrix (see the sensitivity note above — the
    # sample mean is data-dependent and centering by it would leak).
    Sigma = (Xc' * Xc) / nc

    # Analyze-Gauss noise — data in [0,1]^d ⇒ Frobenius sensitivity ≤ d/n
    sens = Float64(d) / nc
    sigma_noise = _rho_to_sigma(rho_cov, sens)

    Sigma_noisy = Sigma + _symmetric_gaussian_noise(d, sigma_noise, rng)

    C = _project_correlation(Sigma_noisy)

    try
        return Copulas.GaussianCopula(C)
    catch e
        @warn "Failed to build Gaussian copula from private covariance " *
              "($(sprint(showerror, e))). Falling back to independent sampling."
        return nothing
    end
end

# ─── _fit_engine(::DPCopulaGenerator, …) ─────────────────────────────────

function _fit_engine(gen::DPCopulaGenerator, cols, col_names, id_set,
                     fill_dict, hints, nm_cache, basetype_cache,
                     nrows, mat, rng, privacy)
    hint_dict   = Dict(h.name => h for h in hints)
    col_kinds   = Symbol[]
    marginals   = Dict{Symbol, DPMarginal}()
    miss        = Dict{Symbol, Float64}()
    copula_cols = Symbol[]
    stat_cols   = Symbol[]

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

        hint = get(hint_dict, name, nothing)
        kind = if hint !== nothing && hint.kind != :identifier
            if hint.levels !== nothing
                observed  = unique(nm)
                uncovered = setdiff(observed, hint.levels)
                if !isempty(uncovered)
                    @warn "ColumnHint for :$name has levels that don't cover " *
                          "observed values: $uncovered"
                end
            end
            hint.kind
        else
            _detect_column_type(nm, T)
        end
        push!(col_kinds, kind)
        push!(stat_cols, name)

        if kind in (:continuous, :integer)
            push!(copula_cols, name)
        end
    end

    # ── Budget allocation ───────────────────────────────────────────────
    rho_total = _eps_delta_to_rho(privacy.epsilon, privacy.delta)
    d_numeric = length(copula_cols)
    d_stat    = length(stat_cols)

    if d_numeric >= 2
        rho_marginals  = rho_total / 2
        rho_covariance = rho_total / 2
    else
        rho_marginals  = rho_total
        rho_covariance = 0.0
    end
    rho_per_marginal = rho_marginals / max(d_stat, 1)

    # ── Fit DP marginals ────────────────────────────────────────────────
    for name in stat_cols
        nm   = nm_cache[name]
        T    = basetype_cache[name]
        kind = col_kinds[findfirst(==(name), col_names)]
        hint = get(hint_dict, name, nothing)
        marginals[name] = _fit_dp_marginal(nm, kind, T, rho_per_marginal, rng;
                                            hint = hint)
    end

    # ── Fit private-covariance Gaussian copula ──────────────────────────
    copula = nothing
    if d_numeric >= 2 && rho_covariance > 0
        copula = _fit_dp_covariance_copula(cols, copula_cols, marginals,
                                            nrows, rho_covariance, rng)
    elseif d_numeric == 1
        @warn "Only one numeric column; copula fitting skipped."
    elseif d_numeric == 0
        @warn "No numeric columns; all columns are categorical/constant. " *
              "Falling back to fully independent sampling."
    end

    id_cols = [name for name in col_names if name in id_set]

    return FittedDPCopulaModel(
        col_names, col_kinds, marginals, miss,
        copula, copula_cols, nrows,
        id_cols, fill_dict, mat, rng,
    )
end
