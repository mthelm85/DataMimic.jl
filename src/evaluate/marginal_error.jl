# ─── pairwise_marginal_error ───────────────────────────────────────────────
#
# REQ-EVL-012: Discretize all columns and compute TVD over every order-way
#              joint distribution
# REQ-EVL-013: Returns NamedTuple with per-pair TVD, mean, worst-case pair
#
# Reference: [McKenna et al. 2019]

const PME_DEFAULT_BINS = 20

# ─── Helpers ────────────────────────────────────────────────────────────────

"""
Discretize a single column into integer bin indices (1-based).
Returns `(bins::Vector{Int}, n_levels::Int)`.
"""
function _discretize_column(col::AbstractVector, n_bins::Int;
                            ref_col::AbstractVector = col)
    kind = _eval_column_kind(ref_col)
    n = length(col)

    if kind == :numeric
        ref_nm = collect(Float64, filter(x -> !ismissing(x) && isfinite(x), ref_col))
        isempty(ref_nm) && return (ones(Int, n), 1)

        lo, hi = extrema(ref_nm)
        lo == hi && return (ones(Int, n), 1)

        bins = Vector{Int}(undef, n)
        for i in 1:n
            v = col[i]
            if ismissing(v) || (v isa AbstractFloat && !isfinite(v))
                bins[i] = 0  # missing → bin 0 (excluded from joint)
            else
                b = clamp(floor(Int, (Float64(v) - lo) / (hi - lo) * n_bins) + 1,
                          1, n_bins)
                bins[i] = b
            end
        end
        return (bins, n_bins)
    else
        # Categorical: map levels to integer codes
        ref_nm = filter(!ismissing, ref_col)
        levels = sort(collect(unique(ref_nm)))
        isempty(levels) && return (ones(Int, n), 1)

        level_map = Dict(lv => i for (i, lv) in enumerate(levels))
        bins = Vector{Int}(undef, n)
        for i in 1:n
            v = col[i]
            bins[i] = ismissing(v) ? 0 : get(level_map, v, 0)
        end
        return (bins, length(levels))
    end
end

"""
Compute TVD between two joint distributions represented as flat count vectors.
"""
function _joint_tvd(r_counts::Vector{Float64}, s_counts::Vector{Float64})
    r_total = sum(r_counts)
    s_total = sum(s_counts)
    (r_total == 0 || s_total == 0) && return 1.0

    d = 0.0
    for i in eachindex(r_counts, s_counts)
        d += abs(r_counts[i] / r_total - s_counts[i] / s_total)
    end
    return d / 2
end

# ─── Public API ─────────────────────────────────────────────────────────────

"""
    pairwise_marginal_error(real, synth; order=2, n_bins=$PME_DEFAULT_BINS) -> NamedTuple

Measure joint distribution error between real and synthetic data by computing
Total Variation Distance over every `order`-way combination of columns.

All columns are discretized: continuous → `n_bins` equal-width bins,
categorical → integer-coded levels.

Returns a `NamedTuple` with:
- `pair_scores`: Dict mapping column tuple → TVD score
- `mean`: arithmetic mean of all pair TVD scores
- `worst_pair`: the column tuple with the highest TVD
- `worst_score`: the highest TVD value
- `n_pairs`: number of column combinations evaluated

All TVD values are in [0, 1] where 0 is identical joint distributions.
"""
function pairwise_marginal_error(real, synth; order::Int = 2,
                                  n_bins::Int = PME_DEFAULT_BINS)
    Tables.istable(real)  || throw(ArgumentError("real must be a Tables.jl table"))
    Tables.istable(synth) || throw(ArgumentError("synth must be a Tables.jl table"))
    order in (2, 3) || throw(ArgumentError("order must be 2 or 3, got $order"))
    n_bins > 0 || throw(ArgumentError("n_bins must be positive, got $n_bins"))

    r_cols = Tables.columns(real)
    s_cols = Tables.columns(synth)
    r_names = Set(Tables.columnnames(r_cols))
    s_names = Set(Tables.columnnames(s_cols))
    shared  = sort(collect(Symbol, intersect(r_names, s_names)))

    length(shared) >= order || throw(ArgumentError(
        "Need at least $order shared columns, got $(length(shared))."))

    # ── Discretize all columns ─────────────────────────────────────────
    n_real  = length(Tables.getcolumn(r_cols, first(shared)))
    n_synth = length(Tables.getcolumn(s_cols, first(shared)))

    r_bins = Dict{Symbol, Vector{Int}}()
    s_bins = Dict{Symbol, Vector{Int}}()
    n_levels = Dict{Symbol, Int}()

    for name in shared
        r_col = Tables.getcolumn(r_cols, name)
        s_col = Tables.getcolumn(s_cols, name)
        rb, nl = _discretize_column(r_col, n_bins; ref_col = r_col)
        sb, _  = _discretize_column(s_col, n_bins; ref_col = r_col)
        r_bins[name]   = rb
        s_bins[name]   = sb
        n_levels[name] = nl
    end

    # ── Enumerate all order-way combinations ───────────────────────────
    combos = collect(_combinations(shared, order))
    pair_scores = Dict{NTuple{order, Symbol}, Float64}()

    for combo in combos
        # Compute joint distribution size
        dims = ntuple(k -> n_levels[combo[k]], Val(order))
        joint_size = prod(dims)

        # Build flat index for both datasets
        r_counts = zeros(Float64, joint_size)
        s_counts = zeros(Float64, joint_size)

        # Count real
        for i in 1:n_real
            skip = false
            idx = 1
            stride = 1
            for k in order:-1:1
                b = r_bins[combo[k]][i]
                if b == 0
                    skip = true
                    break
                end
                idx += (b - 1) * stride
                stride *= dims[k]
            end
            skip || (r_counts[idx] += 1.0)
        end

        # Count synth
        for i in 1:n_synth
            skip = false
            idx = 1
            stride = 1
            for k in order:-1:1
                b = s_bins[combo[k]][i]
                if b == 0
                    skip = true
                    break
                end
                idx += (b - 1) * stride
                stride *= dims[k]
            end
            skip || (s_counts[idx] += 1.0)
        end

        pair_scores[Tuple(combo)] = _joint_tvd(r_counts, s_counts)
    end

    # ── Summarize ──────────────────────────────────────────────────────
    n_pairs = length(pair_scores)
    mean_tvd = n_pairs > 0 ? sum(values(pair_scores)) / n_pairs : 0.0

    worst_pair  = n_pairs > 0 ? argmax(pair_scores) : ntuple(_ -> :none, Val(order))
    worst_score = n_pairs > 0 ? pair_scores[worst_pair] : 0.0

    return (;
        pair_scores  = pair_scores,
        mean         = mean_tvd,
        worst_pair   = worst_pair,
        worst_score  = worst_score,
        n_pairs      = n_pairs,
    )
end

# ─── Combination iterator ─────────────────────────────────────────────────

"""
Generate all `k`-element combinations from `items`.
"""
function _combinations(items::Vector{T}, k::Int) where T
    n = length(items)
    result = NTuple{k, T}[]
    k > n && return result
    _combinations_recurse!(result, items, k, 1, T[])
    return result
end

function _combinations_recurse!(result, items, k, start, current)
    if length(current) == k
        push!(result, Tuple(current))
        return
    end
    for i in start:length(items)
        push!(current, items[i])
        _combinations_recurse!(result, items, k, i + 1, current)
        pop!(current)
    end
end
