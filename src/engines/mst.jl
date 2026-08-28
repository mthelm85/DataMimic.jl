# ─── MSTGenerator Engine ────────────────────────────────────────────────────
#
# MST algorithm [McKenna et al. 2021]:
# 1. Discretize all columns into bins
# 2. Select a spanning tree via the exponential mechanism
# 3. Measure pairwise marginals with calibrated Gaussian noise (zCDP)
# 4. Build conditional distributions P(child | parent) for sampling
#
# REQ-MST-001 through REQ-MST-007.

const MST_DEFAULT_BINS = 32

# ─── Discretization ────────────────────────────────────────────────────────

"""
Discretize the non-missing values `nm` of a single column into integer
bin indices.  Returns `(DiscretizationInfo, bin_idx::Vector{Int})` where
`bin_idx[j]` is the 1-based bin for `nm[j]`.

Continuous/integer columns are binned into `n_bins` equal-width bins.
Categorical/binary columns map each unique level to its own bin.
"""
function _discretize_column(nm::Vector, kind::Symbol, T::Type, n_bins::Int;
                            hint::Union{Nothing, ColumnHint} = nothing)
    # ── Constant ────────────────────────────────────────────────────────
    if kind == :constant
        val = isempty(nm) ? missing : first(nm)
        info = DiscretizationInfo(kind, T, nothing, [val], 1)
        return info, ones(Int, length(nm))
    end

    # ── Continuous / Integer ────────────────────────────────────────────
    if kind in (:continuous, :integer)
        vals = Float64.(nm)
        lo, hi = extrema(vals)
        if lo == hi                       # degenerate — single value
            edges = [lo - 0.5, hi + 0.5]
            info = DiscretizationInfo(kind, T, edges, nothing, 1)
            return info, ones(Int, length(nm))
        end
        edges = collect(range(lo, hi; length = n_bins + 1))
        edges[1]   -= abs(lo) * 1e-10 + 1e-15   # widen so extremes land inside
        edges[end] += abs(hi) * 1e-10 + 1e-15
        bin_idx = [_find_bin(Float64(v), edges) for v in vals]
        info = DiscretizationInfo(kind, T, edges, nothing, n_bins)
        return info, bin_idx
    end

    # ── Categorical / Binary ───────────────────────────────────────────
    lvls = if hint !== nothing && hint.levels !== nothing
        collect(hint.levels)
    else
        try sort(unique(nm)) catch; unique(nm) end
    end
    level_map = Dict(v => i for (i, v) in enumerate(lvls))
    bin_idx = [get(level_map, v, 0) for v in nm]   # 0 = unmapped
    info = DiscretizationInfo(kind, T, nothing, lvls, length(lvls))
    return info, bin_idx
end

# ─── Mutual information ───────────────────────────────────────────────────

"""Mutual information between two discretized columns (pairwise-complete cases)."""
function _mutual_info(col_i::Vector{Int}, col_j::Vector{Int},
                      k_i::Int, k_j::Int, nrows::Int)
    ct = zeros(k_i, k_j)
    nc = 0
    for r in 1:nrows
        @inbounds a, b = col_i[r], col_j[r]
        if a > 0 && b > 0
            ct[a, b] += 1.0
            nc += 1
        end
    end
    nc == 0 && return 0.0

    P  = ct / nc
    pi = vec(sum(P, dims = 2))
    pj = vec(sum(P, dims = 1))

    mi = 0.0
    for a in 1:k_i, b in 1:k_j
        pab = P[a, b]
        if pab > 0 && pi[a] > 0 && pj[b] > 0
            mi += pab * log(pab / (pi[a] * pj[b]))
        end
    end
    return mi
end

# ─── MST tree selection via exponential mechanism ─────────────────────────

"""
Select a spanning tree over `d` discretized columns using Prim's
algorithm with the exponential mechanism for edge selection.
Returns `(tree_edges, root)`.
"""
function _select_mst_tree(disc_data::Vector{Vector{Int}},
                          n_bins::Vector{Int}, nrows::Int,
                          eps_per_step::Float64, rng::AbstractRNG)
    d = length(disc_data)
    d == 1 && return Tuple{Int,Int}[], 1

    # Pre-compute pairwise MI
    MI = zeros(d, d)
    for i in 1:d, j in (i+1):d
        mi = _mutual_info(disc_data[i], disc_data[j],
                          n_bins[i], n_bins[j], nrows)
        MI[i, j] = MI[j, i] = mi
    end

    in_tree = falses(d)
    root = rand(rng, 1:d)
    in_tree[root] = true
    tree_edges = Tuple{Int,Int}[]

    for _ in 1:(d - 1)
        cands  = Tuple{Int,Int}[]
        scores = Float64[]
        for i in 1:d
            in_tree[i] || continue
            for j in 1:d
                in_tree[j] && continue
                push!(cands, (i, j))
                push!(scores, MI[i, j])
            end
        end
        isempty(cands) && break
        sel = _exponential_mechanism(scores, eps_per_step, 1.0, rng)
        parent, child = cands[sel]
        push!(tree_edges, (parent, child))
        in_tree[child] = true
    end

    return tree_edges, root
end

# ─── Noisy measurement + conditional construction ────────────────────────

"""
Measure the root marginal and all pairwise marginals on `tree_edges`
with Gaussian noise (ρ-zCDP per measurement).  Returns
`(root_marginal, conditionals)`.
"""
function _measure_mst(disc_data::Vector{Vector{Int}}, n_bins::Vector{Int},
                      tree_edges::Vector{Tuple{Int,Int}}, root::Int,
                      rho_per::Float64, nrows::Int, rng::AbstractRNG)
    # ── Root 1-way marginal ──
    root_col = disc_data[root]
    kr = n_bins[root]
    counts = zeros(kr)
    for v in root_col
        v > 0 && (counts[v] += 1.0)
    end
    sigma = _rho_to_sigma(rho_per, 1.0)
    counts .+= randn(rng, kr) .* sigma
    counts .= max.(counts, 0.0)
    s = sum(counts)
    root_marginal = s > 0 ? counts / s : fill(1.0 / kr, kr)

    # ── 2-way marginals → P(child | parent) ──
    conditionals = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    for (parent, child) in tree_edges
        col_p, col_c = disc_data[parent], disc_data[child]
        kp, kc = n_bins[parent], n_bins[child]

        ct = zeros(kp, kc)
        for r in 1:nrows
            @inbounds a, b = col_p[r], col_c[r]
            if a > 0 && b > 0
                ct[a, b] += 1.0
            end
        end

        sig = _rho_to_sigma(rho_per, 1.0)
        ct .+= randn(rng, kp, kc) .* sig
        ct .= max.(ct, 0.0)

        cond = similar(ct)
        for i in 1:kp
            rs = sum(@view ct[i, :])
            cond[i, :] = rs > 0 ? ct[i, :] / rs : fill(1.0 / kc, kc)
        end
        conditionals[(parent, child)] = cond
    end

    return root_marginal, conditionals
end

# ─── Undiscretize ────────────────────────────────────────────────────────

"""
Map bin indices back to values in the original domain.

Continuous/integer: uniform sample within the bin.
Categorical/binary/constant: look up the level.
"""
function _undiscretize(bins::Vector{Int}, info::DiscretizationInfo,
                       n::Int, rng::AbstractRNG)
    if info.kind == :constant
        return fill(info.levels[1], n)
    end

    if info.kind in (:continuous, :integer)
        edges = info.bin_edges
        k = length(edges) - 1
        vals = Vector{Float64}(undef, n)
        for i in 1:n
            b = clamp(bins[i], 1, k)
            vals[i] = edges[b] + rand(rng) * (edges[b + 1] - edges[b])
        end
        T = info.original_eltype
        if info.kind == :integer
            return T <: Integer ? round.(T, vals) : round.(vals)
        else
            return T <: AbstractFloat ? convert.(T, vals) : vals
        end
    end

    # Categorical / binary
    lvls = info.levels
    nl = length(lvls)
    return [lvls[clamp(b, 1, nl)] for b in bins]
end

# ─── _fit_engine(::MSTGenerator, …) ─────────────────────────────────────

function _fit_engine(gen::MSTGenerator, cols, col_names, id_set, fill_dict,
                     hints, nm_cache, basetype_cache, nrows, mat, rng, privacy)
    if gen.max_marginal_order == 3
        @warn "3-way marginals are not yet implemented; falling back to 2-way."
    end

    hint_dict  = Dict(h.name => h for h in hints)
    col_kinds  = Symbol[]
    miss       = Dict{Symbol, Float64}()
    stat_cols  = Symbol[]

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
                          "observed values: $uncovered; these will be excluded."
                end
            end
            hint.kind
        else
            _detect_column_type(nm, T)
        end
        push!(col_kinds, kind)
        push!(stat_cols, name)
    end

    d = length(stat_cols)

    # ── Discretize ──────────────────────────────────────────────────────
    disc           = Dict{Symbol, DiscretizationInfo}()
    disc_data_vecs = Vector{Vector{Int}}(undef, d)
    bins_per_col   = Vector{Int}(undef, d)

    for (idx, name) in enumerate(stat_cols)
        nm   = nm_cache[name]
        T    = basetype_cache[name]
        kind = col_kinds[findfirst(==(name), col_names)]
        hint = get(hint_dict, name, nothing)

        info, bin_idx = _discretize_column(nm, kind, T, MST_DEFAULT_BINS;
                                           hint = hint)
        disc[name]         = info
        bins_per_col[idx]  = info.n_bins

        # Map nm bin indices → full-length column (0 = missing / non-finite)
        col  = Tables.getcolumn(cols, name)
        full = zeros(Int, nrows)
        j = 0
        for i in 1:nrows
            v = col[i]
            if !ismissing(v)
                ok = T <: AbstractFloat ? isfinite(Float64(v)) : true
                if ok
                    j += 1
                    full[i] = bin_idx[j]
                end
            end
        end
        disc_data_vecs[idx] = full
    end

    # ── Budget allocation (zCDP) ────────────────────────────────────────
    rho_total   = _eps_delta_to_rho(privacy.epsilon, privacy.delta)
    rho_select  = rho_total / 2
    rho_measure = rho_total / 2

    eps_select   = sqrt(2.0 * rho_select)
    eps_per_step = d > 1 ? eps_select / (d - 1) : eps_select

    n_meas  = max(d, 1)          # 1 root + (d-1) pairwise
    rho_per = rho_measure / n_meas

    # ── Select spanning tree ────────────────────────────────────────────
    tree_edges, root = _select_mst_tree(disc_data_vecs, bins_per_col,
                                         nrows, eps_per_step, rng)

    # ── Noisy measurement ───────────────────────────────────────────────
    root_marginal, conditionals = _measure_mst(disc_data_vecs, bins_per_col,
                                                tree_edges, root,
                                                rho_per, nrows, rng)

    id_cols = [name for name in col_names if name in id_set]

    return FittedMSTModel(
        col_names, col_kinds, stat_cols, disc,
        tree_edges, root, root_marginal, conditionals,
        miss, nrows, id_cols, fill_dict, mat, rng,
    )
end
