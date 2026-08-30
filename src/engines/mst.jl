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

# ─── Noisy 1-way measurement ─────────────────────────────────────────────

"""Noisy 1-way count vector for a discretized column (0 = missing)."""
function _count_oneway_noisy(col::Vector{Int}, k::Int, sigma::Float64,
                             rng::AbstractRNG)
    c = zeros(k)
    for v in col
        v > 0 && (c[v] += 1.0)
    end
    return c .+ randn(rng, k) .* sigma
end

# ─── MST tree selection via exponential mechanism ─────────────────────────

"""
Score a candidate edge by how far the pair's true 2-way marginal sits from what
independence would predict — on the **count** scale, following the reference
implementation.

    q(a, b) = ‖ M_ab(D) − ŷ_a ⊗ ŷ_b / n ‖₁

`ŷ` are the *noisy* 1-way marginals, so the reference point is fixed given the
already-released measurements and the score depends on the data only through
`M_ab`.  Changing one record moves one cell of `M_ab` out and another in, so
the L1 sensitivity is exactly 2.

The count scale is essential and is where the previous mutual-information score
went wrong.  The exponential mechanism weights candidates by `exp(ε·q/(2Δ))`,
so discrimination depends on the *absolute* spread of `q`.  Mutual information
is measured in nats and spans a few tenths regardless of `n`; with
`ε_step ≈ 0.03` on a 15-column table, that put every candidate within
`exp(0.005)` of every other and made selection a uniform random draw.  Scoring
on counts scales the spread with `n`, which is what makes the mechanism able to
tell candidate edges apart at all.
"""
function _edge_scores(disc_data::Vector{Vector{Int}}, n_bins::Vector{Int},
                      oneway_noisy::Vector{Vector{Float64}}, nrows::Int)
    d = length(disc_data)
    S = zeros(d, d)
    for i in 1:d, j in (i + 1):d
        ct  = zeros(n_bins[i], n_bins[j])
        for r in 1:nrows
            @inbounds a, b = disc_data[i][r], disc_data[j][r]
            (a > 0 && b > 0) && (ct[a, b] += 1.0)
        end
        # Independence reference built from the noisy 1-way counts.
        est = (oneway_noisy[i] * oneway_noisy[j]') ./ max(nrows, 1)
        S[i, j] = S[j, i] = sum(abs, ct .- est)
    end
    return S
end

"""
Select a spanning tree over `d` discretized columns using Prim's
algorithm with the exponential mechanism for edge selection.
Returns `(tree_edges, root)`.
"""
function _select_mst_tree(disc_data::Vector{Vector{Int}},
                          n_bins::Vector{Int}, nrows::Int,
                          oneway_noisy::Vector{Vector{Float64}},
                          eps_per_step::Float64, rng::AbstractRNG)
    d = length(disc_data)
    d == 1 && return Tuple{Int,Int}[], 1

    S = _edge_scores(disc_data, n_bins, oneway_noisy, nrows)

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
                push!(scores, S[i, j])
            end
        end
        isempty(cands) && break
        # Sensitivity 2: one record moves one cell of the 2-way marginal out
        # and another in, changing the L1 norm by at most 2.
        sel = _exponential_mechanism(scores, eps_per_step, 2.0, rng)
        parent, child = cands[sel]
        push!(tree_edges, (parent, child))
        in_tree[child] = true
    end

    return tree_edges, root
end

# ─── Private-PGM estimation on the selected tree ─────────────────────────

"""Numerically stable log-sum-exp over a vector."""
function _lse(xs::AbstractVector{Float64})
    m = maximum(xs)
    isfinite(m) || return m
    return m + log(sum(exp(x - m) for x in xs))
end

"""
Exact sum-product belief propagation for a tree-structured Markov random field

    p(x) ∝ exp( Σᵢ θᵢ(xᵢ) + Σ₍ᵢⱼ₎ θᵢⱼ(xᵢ, xⱼ) )

Because selection produces a *spanning tree*, the model is a tree and inference
is exact in two passes (leaves→root, root→leaves) — no junction-tree machinery
over general cliques is required.  Returns node and edge marginals, each
normalized to sum to 1.  All work is in log space.
"""
function _tree_bp(edges::Vector{Tuple{Int,Int}}, nbrs::Vector{Vector{Int}},
                  root::Int, n_bins::Vector{Int},
                  θ_node::Vector{Vector{Float64}},
                  θ_edge::Dict{Tuple{Int,Int}, Matrix{Float64}})
    d = length(n_bins)

    order  = [root]
    parent = zeros(Int, d)
    seen   = falses(d); seen[root] = true
    qi = 1
    while qi <= length(order)
        u = order[qi]; qi += 1
        for v in nbrs[u]
            if !seen[v]
                seen[v] = true
                parent[v] = u
                push!(order, v)
            end
        end
    end

    logm = Dict{Tuple{Int,Int}, Vector{Float64}}()
    pot(p, c) = haskey(θ_edge, (p, c)) ? θ_edge[(p, c)] : permutedims(θ_edge[(c, p)])

    # Upward: children → parents
    for idx in length(order):-1:2
        c = order[idx]; p = parent[c]
        belief_c = copy(θ_node[c])
        for g in nbrs[c]
            g == p && continue
            belief_c .+= logm[(g, c)]
        end
        E = pot(p, c)
        logm[(c, p)] = [_lse(vec(E[xp, :]) .+ belief_c) for xp in 1:n_bins[p]]
    end

    # Downward: parents → children
    for u in order
        for c in nbrs[u]
            c == parent[u] && continue
            belief_u = copy(θ_node[u])
            for k in nbrs[u]
                k == c && continue
                belief_u .+= logm[(k, u)]
            end
            E = pot(u, c)
            logm[(u, c)] = [_lse(vec(E[:, xc]) .+ belief_u) for xc in 1:n_bins[c]]
        end
    end

    μ_node = Vector{Vector{Float64}}(undef, d)
    for i in 1:d
        lb = copy(θ_node[i])
        for k in nbrs[i]
            lb .+= logm[(k, i)]
        end
        lb .-= _lse(lb)
        μ_node[i] = exp.(lb)
    end

    μ_edge = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    for (p, c) in edges
        up = copy(θ_node[p]); for k in nbrs[p]; k == c && continue; up .+= logm[(k, p)]; end
        uc = copy(θ_node[c]); for k in nbrs[c]; k == p && continue; uc .+= logm[(k, c)]; end
        E = pot(p, c)
        lb = Matrix{Float64}(undef, n_bins[p], n_bins[c])
        for xp in 1:n_bins[p], xc in 1:n_bins[c]
            lb[xp, xc] = up[xp] + uc[xc] + E[xp, xc]
        end
        lb .-= _lse(vec(lb))
        μ_edge[(p, c)] = exp.(lb)
    end

    return μ_node, μ_edge
end

"""
Fit a tree-structured MRF to the noisy measurements [McKenna et al. 2019].

Minimizes `Σ‖μ − y‖²` over the marginal polytope by entropic mirror descent:
each iteration computes the model's marginals by belief propagation, takes the
gradient of the loss with respect to them, and subtracts it from the
potentials.  Because θ ↦ μ is the gradient of the log-partition function, that
update *is* mirror descent under the entropy mirror map.

This is what makes the measurements mutually consistent; without it the 2-way
counts are merely row-normalized and the 1-way measurements are discarded.

Targets are on the probability scale and deliberately not clamped: noise can
push a cell below zero, and least squares against the raw value is unbiased
where clamping is not.  The fitted marginals are valid probabilities by
construction.  Estimation is post-processing of already-private measurements
and consumes no additional privacy budget.
"""
function _fit_tree_mrf(edges::Vector{Tuple{Int,Int}}, nbrs::Vector{Vector{Int}},
                       root::Int, n_bins::Vector{Int},
                       y_node::Vector{Vector{Float64}},
                       y_edge::Dict{Tuple{Int,Int}, Matrix{Float64}};
                       iters::Int = 250, lr::Float64 = 1.0)
    d = length(n_bins)
    θ_node = [zeros(n_bins[i]) for i in 1:d]
    θ_edge = Dict{Tuple{Int,Int}, Matrix{Float64}}(
        e => zeros(n_bins[e[1]], n_bins[e[2]]) for e in edges)

    objective(mn, me) =
        sum(sum(abs2, mn[i] .- y_node[i]) for i in 1:d) +
        sum(sum(abs2, me[e] .- y_edge[e]) for e in edges; init = 0.0)

    μ_node, μ_edge = _tree_bp(edges, nbrs, root, n_bins, θ_node, θ_edge)
    loss = objective(μ_node, μ_edge)

    step = lr
    for _ in 1:iters
        g_node = [2.0 .* (μ_node[i] .- y_node[i]) for i in 1:d]
        g_edge = Dict{Tuple{Int,Int}, Matrix{Float64}}(
            e => 2.0 .* (μ_edge[e] .- y_edge[e]) for e in edges)

        accepted = false
        for _ in 1:40
            θn = [θ_node[i] .- step .* g_node[i] for i in 1:d]
            θe = Dict{Tuple{Int,Int}, Matrix{Float64}}(
                e => θ_edge[e] .- step .* g_edge[e] for e in edges)
            mn, me = _tree_bp(edges, nbrs, root, n_bins, θn, θe)
            l = objective(mn, me)
            if l < loss
                θ_node, θ_edge, μ_node, μ_edge, loss = θn, θe, mn, me, l
                accepted = true
                break
            end
            step /= 2
        end
        accepted || break
        step *= 1.5
    end

    return μ_node, μ_edge
end

# ─── Noisy measurement + conditional construction ────────────────────────

"""
Measure the root marginal and all pairwise marginals on `tree_edges`
with Gaussian noise (ρ-zCDP per measurement).  Returns
`(root_marginal, conditionals)`.
"""
function _measure_mst(disc_data::Vector{Vector{Int}}, n_bins::Vector{Int},
                      tree_edges::Vector{Tuple{Int,Int}}, root::Int,
                      oneway_noisy::Vector{Vector{Float64}},
                      rho_per::Float64, nrows::Int, rng::AbstractRNG;
                      reconcile::Bool = true)
    d = length(disc_data)
    sigma = _rho_to_sigma(rho_per, 1.0)

    # ── 2-way marginals on the selected edges ──
    raw2 = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    for (parent, child) in tree_edges
        col_p, col_c = disc_data[parent], disc_data[child]
        kp, kc = n_bins[parent], n_bins[child]
        ct = zeros(kp, kc)
        for r in 1:nrows
            @inbounds a, b = col_p[r], col_c[r]
            (a > 0 && b > 0) && (ct[a, b] += 1.0)
        end
        ct .+= randn(rng, kp, kc) .* sigma
        raw2[(parent, child)] = ct
    end

    if reconcile && !isempty(tree_edges)
        # ── Private-PGM: reconcile the 1-way and 2-way measurements ──
        nbrs = [Int[] for _ in 1:d]
        for (p, c) in tree_edges
            push!(nbrs[p], c)
            push!(nbrs[c], p)
        end
        n = max(nrows, 1)
        y_node = [v ./ n for v in oneway_noisy]
        y_edge = Dict{Tuple{Int,Int}, Matrix{Float64}}(
            e => m ./ n for (e, m) in raw2)

        μ_node, μ_edge = _fit_tree_mrf(tree_edges, nbrs, root, n_bins,
                                       y_node, y_edge)

        rm = copy(μ_node[root])
        s = sum(rm)
        root_marginal = s > 0 ? rm ./ s : fill(1.0 / length(rm), length(rm))

        conditionals = Dict{Tuple{Int,Int}, Matrix{Float64}}()
        for (p, c) in tree_edges
            joint = μ_edge[(p, c)]
            kp, kc = size(joint)
            cond = Matrix{Float64}(undef, kp, kc)
            for i in 1:kp
                r = sum(@view joint[i, :])
                cond[i, :] = r > 0 ? joint[i, :] ./ r : fill(1.0 / kc, kc)
            end
            conditionals[(p, c)] = cond
        end
        return root_marginal, conditionals
    end

    # ── Unreconciled: clamp and row-normalize each measurement on its own ──
    kr = n_bins[root]
    counts = max.(oneway_noisy[root], 0.0)
    s = sum(counts)
    root_marginal = s > 0 ? counts / s : fill(1.0 / kr, kr)

    conditionals = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    for (parent, child) in tree_edges
        ct = max.(raw2[(parent, child)], 0.0)
        kp, kc = size(ct)
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
    #
    # Three ways, as in [McKenna et al. 2021]: selection, the 1-way marginals
    # that anchor the selection score, and the 2-way marginals on the chosen
    # edges.  zCDP composes additively across all of them.
    # The 1-way marginals serve only as the independence reference for the
    # selection score, so they are given a small share: a coarse reference is
    # enough to rank candidate edges, and the budget is far more valuable in
    # the 2-way marginals that become the sampling conditionals.
    rho_total   = _eps_delta_to_rho(privacy.epsilon, privacy.delta)
    rho_select  = 0.30 * rho_total
    rho_oneway  = 0.20 * rho_total
    rho_measure = 0.50 * rho_total

    # Exponential mechanism costs ε²/8 in zCDP [Bun & Steinke 2016, Prop. 3].
    # With (d-1) sequential selections: (d-1)·(ε_step)²/8 = ρ_select
    #   ⟹  ε_step = √(8·ρ_select/(d-1))
    eps_per_step = d > 1 ? sqrt(8.0 * rho_select / (d - 1)) : sqrt(8.0 * rho_select)

    # ── Measure 1-way marginals (anchor for the selection score) ────────
    sigma_oneway = _rho_to_sigma(rho_oneway / max(d, 1), 1.0)
    oneway_noisy = [_count_oneway_noisy(disc_data_vecs[i], bins_per_col[i],
                                        sigma_oneway, rng) for i in 1:d]

    n_meas  = max(d, 1)          # 1 root + (d-1) pairwise
    rho_per = rho_measure / n_meas

    # ── Select spanning tree ────────────────────────────────────────────
    tree_edges, root = _select_mst_tree(disc_data_vecs, bins_per_col,
                                         nrows, oneway_noisy, eps_per_step, rng)

    # ── Noisy measurement ───────────────────────────────────────────────
    root_marginal, conditionals = _measure_mst(disc_data_vecs, bins_per_col,
                                                tree_edges, root, oneway_noisy,
                                                rho_per, nrows, rng)

    id_cols = [name for name in col_names if name in id_set]

    return FittedMSTModel(
        col_names, col_kinds, stat_cols, disc,
        tree_edges, root, root_marginal, conditionals,
        miss, nrows, id_cols, fill_dict, mat, rng,
    )
end
