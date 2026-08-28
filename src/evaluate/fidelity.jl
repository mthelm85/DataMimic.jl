# ─── fidelity_score ─────────────────────────────────────────────────────────
#
# REQ-EVL-001: KS statistics for continuous columns
# REQ-EVL-002: Total Variation Distance for categorical columns
# REQ-EVL-003: Frobenius norm of Spearman correlation difference
# REQ-EVL-004: Returns NamedTuple with per-column, 2D, and aggregate scores

# ─── Helpers ────────────────────────────────────────────────────────────────

"""
Kolmogorov–Smirnov statistic: max |F_real(x) − F_synth(x)|.
Lower is better (0 = identical distributions).
"""
function _ks_statistic(real::AbstractVector, synth::AbstractVector)
    r = filter(x -> !ismissing(x) && isfinite(x), real)
    s = filter(x -> !ismissing(x) && isfinite(x), synth)
    (isempty(r) || isempty(s)) && return 1.0

    r_sorted = sort(collect(Float64, r))
    s_sorted = sort(collect(Float64, s))
    nr = length(r_sorted)
    ns = length(s_sorted)

    # Merge both sorted arrays and walk the ECDFs
    ks = 0.0
    ir, is = 1, 1
    fr, fs = 0.0, 0.0
    while ir ≤ nr || is ≤ ns
        # Advance both sides when values tie (avoids 1/N artefact)
        if ir ≤ nr && is ≤ ns && r_sorted[ir] == s_sorted[is]
            fr = ir / nr; ir += 1
            fs = is / ns; is += 1
        elseif is > ns || (ir ≤ nr && r_sorted[ir] < s_sorted[is])
            fr = ir / nr; ir += 1
        else
            fs = is / ns; is += 1
        end
        ks = max(ks, abs(fr - fs))
    end
    return ks
end

"""
Total Variation Distance: 0.5 Σ|p − q|.
Lower is better (0 = identical distributions).
"""
function _tvd(real::AbstractVector, synth::AbstractVector)
    r = filter(!ismissing, real)
    s = filter(!ismissing, synth)
    (isempty(r) || isempty(s)) && return 1.0

    # Frequency maps
    levels = union(unique(r), unique(s))
    nr, ns = length(r), length(s)
    d = 0.0
    for lv in levels
        p = count(==(lv), r) / nr
        q = count(==(lv), s) / ns
        d += abs(p - q)
    end
    return d / 2
end

"""
Classify a column as `:numeric` or `:categorical` for evaluation purposes.
"""
function _eval_column_kind(col::AbstractVector)
    nm = filter(!ismissing, col)
    isempty(nm) && return :categorical
    T = typeof(first(nm))
    T <: Number ? :numeric : :categorical
end

"""
Spearman correlation matrix for the given named numeric columns.
Missing/non-finite values are replaced with column medians.
"""
function _spearman_corr_matrix(cols, names::Vector{Symbol})
    d = length(names)
    d == 0 && return zeros(0, 0)

    # Rank each column (ties → midrank)
    ranked = Matrix{Float64}(undef, 0, d)
    n = 0
    for (j, name) in enumerate(names)
        raw = collect(Float64, filter(x -> !ismissing(x) && isfinite(x),
                                       Tables.getcolumn(cols, name)))
        if j == 1
            n = length(raw)
            ranked = Matrix{Float64}(undef, n, d)
        end
        # Pad / truncate to match n (defensive)
        if length(raw) < n
            append!(raw, fill(isempty(raw) ? 0.0 : StatsBase.median(raw),
                              n - length(raw)))
        elseif length(raw) > n
            raw = raw[1:n]
        end
        ranked[:, j] = StatsBase.competerank(raw)  # average tie-breaking
    end

    n ≤ 1 && return LinearAlgebra.I(d) |> Matrix{Float64}
    return StatsBase.cor(ranked)
end

# ─── Public API ─────────────────────────────────────────────────────────────

"""
    fidelity_score(real, synth) -> NamedTuple

Evaluate distributional fidelity of synthetic data vs. real data.

Returns a `NamedTuple` with:
- `column_scores`: Dict mapping column name → per-column score (KS or TVD)
- `column_metrics`: Dict mapping column name → metric name (`:ks` or `:tvd`)
- `correlation_score`: Frobenius norm of Spearman correlation difference
- `aggregate`: weighted mean of per-column mean and correlation score

All scores are in [0, 1] where 0 is perfect fidelity.
"""
function fidelity_score(real, synth)
    Tables.istable(real)  || throw(ArgumentError("real must be a Tables.jl table"))
    Tables.istable(synth) || throw(ArgumentError("synth must be a Tables.jl table"))

    r_cols = Tables.columns(real)
    s_cols = Tables.columns(synth)
    r_names = Set(Tables.columnnames(r_cols))
    s_names = Set(Tables.columnnames(s_cols))
    shared  = sort(collect(Symbol, intersect(r_names, s_names)))

    isempty(shared) && throw(ArgumentError(
        "real and synth share no column names."))

    # ── 1D per-column scores ────────────────────────────────────────────
    col_scores  = Dict{Symbol, Float64}()
    col_metrics = Dict{Symbol, Symbol}()
    numeric_cols = Symbol[]

    for name in shared
        r_col = Tables.getcolumn(r_cols, name)
        s_col = Tables.getcolumn(s_cols, name)
        kind  = _eval_column_kind(r_col)

        if kind == :numeric
            col_scores[name]  = _ks_statistic(r_col, s_col)
            col_metrics[name] = :ks
            push!(numeric_cols, name)
        else
            col_scores[name]  = _tvd(r_col, s_col)
            col_metrics[name] = :tvd
        end
    end

    # ── 2D Spearman correlation ─────────────────────────────────────────
    corr_score = if length(numeric_cols) ≥ 2
        R_real  = _spearman_corr_matrix(r_cols, numeric_cols)
        R_synth = _spearman_corr_matrix(s_cols, numeric_cols)
        # Normalize Frobenius by matrix size so score is in [0, 1] range
        d = length(numeric_cols)
        raw = LinearAlgebra.norm(R_real - R_synth)  # Frobenius norm
        clamp(raw / d, 0.0, 1.0)   # scale to ≈[0,1]
    else
        0.0  # no correlation to compare
    end

    # ── Aggregate ───────────────────────────────────────────────────────
    mean_1d = isempty(col_scores) ? 0.0 :
              sum(values(col_scores)) / length(col_scores)
    # Equal weight for 1D mean and 2D correlation
    aggregate = if length(numeric_cols) ≥ 2
        0.5 * mean_1d + 0.5 * corr_score
    else
        mean_1d
    end

    return (;
        column_scores    = col_scores,
        column_metrics   = col_metrics,
        correlation_score = corr_score,
        aggregate        = aggregate,
    )
end
