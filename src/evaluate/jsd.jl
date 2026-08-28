# ─── jensen_shannon ────────────────────────────────────────────────────────
#
# REQ-EVL-010: Per-column Jensen–Shannon divergence
# REQ-EVL-011: Returns NamedTuple with per-column JSD, mean, aggregate
#
# Reference: [Lin 1991] — JSD is symmetric, bounded [0, log(2)]

# ─── Helpers ────────────────────────────────────────────────────────────────

"""
KL divergence: Σ p * log(p / q). Handles zeros via convention 0*log(0/q) = 0.
"""
function _kl_divergence(p::Vector{Float64}, q::Vector{Float64})
    kl = 0.0
    for i in eachindex(p, q)
        p[i] > 0 && q[i] > 0 && (kl += p[i] * log(p[i] / q[i]))
    end
    return kl
end

"""
Jensen–Shannon divergence between two probability vectors.
JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5*(P + Q).
Bounded in [0, log(2)].
"""
function _jsd(p::Vector{Float64}, q::Vector{Float64})
    m = 0.5 .* (p .+ q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
end

"""
Discretize a numeric column into `n_bins` equal-width bins.
Returns a probability vector of length `n_bins`.
"""
function _discretize_to_probs(vals::Vector{Float64}, n_bins::Int;
                               lo::Float64 = NaN, hi::Float64 = NaN)
    isempty(vals) && return fill(1.0 / n_bins, n_bins)

    if isnan(lo) || isnan(hi)
        lo = minimum(vals)
        hi = maximum(vals)
    end

    # Handle constant columns
    if lo == hi
        probs = zeros(n_bins)
        probs[1] = 1.0
        return probs
    end

    counts = zeros(n_bins)
    for v in vals
        b = clamp(floor(Int, (v - lo) / (hi - lo) * n_bins) + 1, 1, n_bins)
        counts[b] += 1.0
    end
    s = sum(counts)
    return s > 0 ? counts / s : fill(1.0 / n_bins, n_bins)
end

"""
Compute probability vector from categorical values given a shared level set.
"""
function _categorical_probs(vals::AbstractVector, levels::Vector)
    n = length(vals)
    n == 0 && return fill(1.0 / max(length(levels), 1), max(length(levels), 1))
    counts = zeros(length(levels))
    level_idx = Dict(lv => i for (i, lv) in enumerate(levels))
    for v in vals
        idx = get(level_idx, v, 0)
        idx > 0 && (counts[idx] += 1.0)
    end
    s = sum(counts)
    return s > 0 ? counts / s : fill(1.0 / length(levels), length(levels))
end

# ─── Public API ─────────────────────────────────────────────────────────────

"""
    jensen_shannon(real, synth; n_bins=50) -> NamedTuple

Compute per-column Jensen–Shannon divergence between real and synthetic data.

Continuous columns are discretized into `n_bins` equal-width bins (using the
range from `real`). Categorical columns use observed level frequencies.

Returns a `NamedTuple` with:
- `column_scores`: Dict mapping column name → JSD value
- `column_kinds`: Dict mapping column name → `:numeric` or `:categorical`
- `mean`: arithmetic mean of per-column JSD values
- `aggregate`: same as `mean` (no 2D component for JSD)

All JSD values are in [0, log(2)] ≈ [0, 0.693] where 0 is identical.
"""
function jensen_shannon(real, synth; n_bins::Int = 50)
    Tables.istable(real)  || throw(ArgumentError("real must be a Tables.jl table"))
    Tables.istable(synth) || throw(ArgumentError("synth must be a Tables.jl table"))
    n_bins > 0 || throw(ArgumentError("n_bins must be positive, got $n_bins"))

    r_cols = Tables.columns(real)
    s_cols = Tables.columns(synth)
    r_names = Set(Tables.columnnames(r_cols))
    s_names = Set(Tables.columnnames(s_cols))
    shared  = sort(collect(Symbol, intersect(r_names, s_names)))

    isempty(shared) && throw(ArgumentError(
        "real and synth share no column names."))

    col_scores = Dict{Symbol, Float64}()
    col_kinds  = Dict{Symbol, Symbol}()

    for name in shared
        r_col = Tables.getcolumn(r_cols, name)
        s_col = Tables.getcolumn(s_cols, name)
        kind  = _eval_column_kind(r_col)

        if kind == :numeric
            r_nm = collect(Float64, filter(x -> !ismissing(x) && isfinite(x), r_col))
            s_nm = collect(Float64, filter(x -> !ismissing(x) && isfinite(x), s_col))

            if isempty(r_nm) || isempty(s_nm)
                col_scores[name] = log(2)  # max divergence
            else
                # Use real data range for consistent binning
                lo, hi = extrema(r_nm)
                # Extend range slightly to include synth values
                if !isempty(s_nm)
                    lo = min(lo, minimum(s_nm))
                    hi = max(hi, maximum(s_nm))
                end
                p = _discretize_to_probs(r_nm, n_bins; lo = lo, hi = hi)
                q = _discretize_to_probs(s_nm, n_bins; lo = lo, hi = hi)
                col_scores[name] = _jsd(p, q)
            end
            col_kinds[name] = :numeric
        else
            r_nm = filter(!ismissing, r_col)
            s_nm = filter(!ismissing, s_col)

            if isempty(r_nm) || isempty(s_nm)
                col_scores[name] = log(2)
            else
                levels = sort(collect(union(unique(r_nm), unique(s_nm))))
                p = _categorical_probs(r_nm, levels)
                q = _categorical_probs(s_nm, levels)
                col_scores[name] = _jsd(p, q)
            end
            col_kinds[name] = :categorical
        end
    end

    mean_jsd = isempty(col_scores) ? 0.0 :
               sum(values(col_scores)) / length(col_scores)

    return (;
        column_scores = col_scores,
        column_kinds  = col_kinds,
        mean          = mean_jsd,
        aggregate     = mean_jsd,
    )
end
