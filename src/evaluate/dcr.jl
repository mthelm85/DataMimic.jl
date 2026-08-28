# ─── privacy_dcr ───────────────────────────────────────────────────────────
#
# REQ-EVL-005: DCR for every synthetic row
# REQ-EVL-006: Returns DCR vector, median, 5th percentile, exact match count
#
# Reference: [Zhao et al. 2021] — CTAB-GAN, DCR metric

# ─── Helpers ────────────────────────────────────────────────────────────────

"""
Encode a table into a normalized numeric matrix for distance computation.

- Numeric columns → min-max scaled to [0, 1] using real data's range.
- Categorical columns → integer codes, then scaled to [0, 1].
- Missing values → column median (numeric) or mode code (categorical).

Returns `(matrix, col_ranges)` where matrix is `(n_rows, n_features)`.
"""
function _encode_for_distance(cols, col_names::Vector{Symbol},
                              ref_cols  # reference table for scaling params
                             )
    nrows = length(Tables.getcolumn(cols, first(col_names)))
    features = Vector{Vector{Float64}}()

    for name in col_names
        raw = Tables.getcolumn(cols, name)
        ref = Tables.getcolumn(ref_cols, name)
        kind = _eval_column_kind(ref)

        if kind == :numeric
            # Min-max normalization using reference (real) data range
            ref_nm = collect(Float64, filter(x -> !ismissing(x) && isfinite(x), ref))
            if isempty(ref_nm)
                push!(features, zeros(nrows))
                continue
            end
            lo, hi = extrema(ref_nm)
            span = hi - lo
            span = span > 0 ? span : 1.0
            med = StatsBase.median(ref_nm)

            col_f = Vector{Float64}(undef, nrows)
            for i in 1:nrows
                v = raw[i]
                if ismissing(v) || !isfinite(v)
                    col_f[i] = (med - lo) / span
                else
                    col_f[i] = (Float64(v) - lo) / span
                end
            end
            push!(features, col_f)
        else
            # Categorical: encode as integers, then scale
            ref_nm = filter(!ismissing, ref)
            levels = unique(ref_nm)
            level_map = Dict(lv => Float64(i) for (i, lv) in enumerate(levels))
            K = Float64(max(length(levels), 1))
            mode_val = if isempty(ref_nm)
                0.0
            else
                # Mode: most frequent level's code
                level_map[StatsBase.mode(ref_nm)]
            end

            col_f = Vector{Float64}(undef, nrows)
            for i in 1:nrows
                v = raw[i]
                if ismissing(v)
                    col_f[i] = mode_val / K
                else
                    col_f[i] = get(level_map, v, 0.0) / K
                end
            end
            push!(features, col_f)
        end
    end

    # Stack into (nrows, n_features) matrix
    mat = Matrix{Float64}(undef, nrows, length(features))
    for (j, f) in enumerate(features)
        mat[:, j] = f
    end
    return mat
end

"""
Compute L2 distance between row `i` of A and row `j` of B.
"""
@inline function _row_dist(A::Matrix{Float64}, i::Int,
                            B::Matrix{Float64}, j::Int)
    d = length(A[i, :])
    s = 0.0
    @inbounds for k in 1:d
        δ = A[i, k] - B[j, k]
        s += δ * δ
    end
    return sqrt(s)
end

# ─── Public API ─────────────────────────────────────────────────────────────

"""
    privacy_dcr(real, synth) -> NamedTuple

Compute the Distance to Closest Record (DCR) for every synthetic row.

For each synthetic row, DCR is the L2 distance (after min-max normalization)
to the nearest real row. Low DCR values indicate potential memorization.

Returns a `NamedTuple` with:
- `dcr`: Vector of DCR values (one per synthetic row)
- `median`: median DCR
- `p5`: 5th percentile of DCR
- `exact_matches`: count of synthetic rows with DCR = 0
"""
function privacy_dcr(real, synth)
    Tables.istable(real)  || throw(ArgumentError("real must be a Tables.jl table"))
    Tables.istable(synth) || throw(ArgumentError("synth must be a Tables.jl table"))

    r_cols = Tables.columns(real)
    s_cols = Tables.columns(synth)
    r_names = Set(Tables.columnnames(r_cols))
    s_names = Set(Tables.columnnames(s_cols))
    shared  = sort(collect(Symbol, intersect(r_names, s_names)))

    isempty(shared) && throw(ArgumentError(
        "real and synth share no column names."))

    # Encode both tables using real data's scale
    R = _encode_for_distance(r_cols, shared, r_cols)
    S = _encode_for_distance(s_cols, shared, r_cols)

    n_real  = size(R, 1)
    n_synth = size(S, 1)

    # Compute DCR for each synthetic row
    dcr = Vector{Float64}(undef, n_synth)
    @inbounds for i in 1:n_synth
        min_d = Inf
        for j in 1:n_real
            d = _row_dist(S, i, R, j)
            d < min_d && (min_d = d)
        end
        dcr[i] = min_d
    end

    med     = StatsBase.median(dcr)
    p5      = StatsBase.percentile(dcr, 5)
    matches = count(==(0.0), dcr)

    return (;
        dcr            = dcr,
        median         = med,
        p5             = p5,
        exact_matches  = matches,
    )
end
