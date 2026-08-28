# ─── DataMimicLuxExt ────────────────────────────────────────────────────────
#
# Package extension: DiffusionGenerator engine (TabDDPM).
#
# Loaded when both Lux.jl and Zygote.jl are present.
# Implements REQ-DIF-001 through REQ-DIF-009.
#
# References:
#   [Kotelnikov et al. 2023] — TabDDPM architecture
#   [Ho et al. 2020]         — DDPM (Gaussian diffusion)
#   [Hoogeboom et al. 2021]  — Multinomial diffusion
#   [Abadi et al. 2016]      — DP-SGD
#   [Mironov 2017]           — Rényi DP accounting

module DataMimicLuxExt

using DataMimic
using DataMimic: _nonmissing, _basetype, _detect_column_type, _postprocess,
                 ConstantMarginal, CategoricalMarginal, _sample_categorical,
                 _eps_delta_to_rho, _rho_to_sigma,
                 FittedDiffusionModel, DiffusionGenerator, PrivacyBudget,
                 ColumnHint, AbstractFittedModel
import DataMimic: _fit_engine, sample
using Lux
import Lux.Training
import Optimisers
import Zygote
using Random: AbstractRNG
import Random
import Tables
import StatsBase
import LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

"""
Pack table columns into Float32 arrays for training.

Returns `(X_num, X_cat_onehot, cat_indices, preprocess_info)`.
- `X_num`: `(d_num, N)` Float32 matrix, z-score normalized.
- `X_cat_onehot`: `(sum(cat_dims), N)` Float32 matrix.
- `cat_indices`: `(n_cat, N)` Int matrix — original category index per row.
"""
function _preprocess(cols, col_names, col_kinds, id_set, hints, nrows)
    hint_dict = Dict(h.name => h for h in hints)

    num_cols   = Symbol[]
    cat_cols   = Symbol[]
    num_means  = Float32[]
    num_stds   = Float32[]
    cat_levels = Dict{Symbol, Vector}()
    cat_dims   = Int[]

    for (i, name) in enumerate(col_names)
        kind = col_kinds[i]
        kind == :identifier && continue

        if kind in (:continuous, :integer)
            push!(num_cols, name)
            nm = Float32.(DataMimic._nonmissing(collect(Tables.getcolumn(cols, name))))
            μ  = length(nm) > 0 ? sum(nm) / length(nm) : 0f0
            σ  = length(nm) > 1 ? sqrt(sum((nm .- μ).^2) / (length(nm) - 1)) : 1f0
            σ  = max(σ, 1f-7)
            push!(num_means, μ)
            push!(num_stds, σ)
        elseif kind in (:categorical, :binary)
            push!(cat_cols, name)
            hint = get(hint_dict, name, nothing)
            nm   = DataMimic._nonmissing(collect(Tables.getcolumn(cols, name)))
            lvls = if hint !== nothing && hint.levels !== nothing
                hint.levels
            else
                sort(unique(nm))
            end
            cat_levels[name] = lvls
            push!(cat_dims, length(lvls))
        end
        # :constant columns are skipped in the neural model
    end

    d_num = length(num_cols)
    d_cat_total = sum(cat_dims; init = 0)

    # ── Build numeric matrix ────────────────────────────────────────────
    X_num = zeros(Float32, d_num, nrows)
    for (j, name) in enumerate(num_cols)
        col = Tables.getcolumn(cols, name)
        μ, σ = num_means[j], num_stds[j]
        for i in 1:nrows
            v = col[i]
            X_num[j, i] = ismissing(v) ? 0f0 : (Float32(v) - μ) / σ
        end
    end

    # ── Build categorical one-hot + index matrices ──────────────────────
    X_cat_oh = zeros(Float32, d_cat_total, nrows)
    cat_indices = zeros(Int, length(cat_cols), nrows)
    offset = 0
    for (c, name) in enumerate(cat_cols)
        lvls = cat_levels[name]
        K = cat_dims[c]
        col = Tables.getcolumn(cols, name)
        lvl_map = Dict(v => k for (k, v) in enumerate(lvls))
        for i in 1:nrows
            v = col[i]
            k = get(lvl_map, ismissing(v) ? first(lvls) : v, 1)
            X_cat_oh[offset + k, i] = 1f0
            cat_indices[c, i] = k
        end
        offset += K
    end

    info = (; num_cols, cat_cols, num_means, num_stds,
              cat_levels, cat_dims, d_num, d_cat_total)
    return X_num, X_cat_oh, cat_indices, info
end

# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise Schedule
# ═══════════════════════════════════════════════════════════════════════════

"""Linear β schedule [Ho et al. 2020]: β_1 = 1e-4, β_T = 0.02."""
function _linear_schedule(T::Int)
    betas = Float32.(range(1f-4, 0.02f0; length = T))
    alphas = 1f0 .- betas
    alphas_cumprod = cumprod(alphas)
    return betas, alphas_cumprod
end

# ═══════════════════════════════════════════════════════════════════════════
# 3. Sinusoidal Timestep Embedding (Lux layer)
# ═══════════════════════════════════════════════════════════════════════════

"""
Custom Lux layer: sinusoidal positional embedding for diffusion timestep.

Maps integer timestep `t` ∈ {1,…,T} to a `dim`-dimensional embedding vector.
"""
struct SinusoidalEmbedding <: Lux.AbstractLuxLayer
    dim::Int
end

function Lux.initialparameters(::AbstractRNG, ::SinusoidalEmbedding)
    return NamedTuple()
end

function Lux.initialstates(::AbstractRNG, layer::SinusoidalEmbedding)
    half = layer.dim ÷ 2
    freqs = Float32.(exp.(-log(10000f0) .* (0:(half - 1)) ./ half))
    return (; freqs = freqs)
end

function (layer::SinusoidalEmbedding)(t, ps, st)
    # t: (1, batch) or (batch,) — integer timesteps
    t_flat = Float32.(vec(t))       # (batch,)
    freqs  = st.freqs               # (half,)
    args   = t_flat' .* freqs       # (half, batch)
    emb    = vcat(sin.(args), cos.(args))  # (dim, batch)
    return emb, st
end

Lux.statelength(l::SinusoidalEmbedding) = l.dim ÷ 2

# ═══════════════════════════════════════════════════════════════════════════
# 4. ResNet MLP Backbone
# ═══════════════════════════════════════════════════════════════════════════

"""
Build the TabDDPM denoising network.

Architecture: input projection → [ResNet block × n_blocks] → output heads.

- `d_in`:  concatenated input dimension (d_num + d_cat_onehot + embed_dim)
- `d_num`: number of numeric output channels (predict noise ε)
- `cat_dims`: per-categorical output logits
- `hidden`: hidden dimension
- `n_blocks`: number of residual blocks
- `embed_dim`: timestep embedding dimension
"""
function _build_model(d_in::Int, d_num::Int, cat_dims::Vector{Int};
                      hidden::Int = 256, n_blocks::Int = 4,
                      embed_dim::Int = 128, dropout::Float64 = 0.0)
    d_out = d_num + sum(cat_dims; init = 0)

    layers = []

    # Input projection
    push!(layers, Dense(d_in => hidden, relu))

    # Residual blocks
    for _ in 1:n_blocks
        block = if dropout > 0
            SkipConnection(
                Chain(Dense(hidden => hidden, relu),
                      Lux.Dropout(Float32(dropout)),
                      Dense(hidden => hidden)),
                +)
        else
            SkipConnection(
                Chain(Dense(hidden => hidden, relu),
                      Dense(hidden => hidden)),
                +)
        end
        push!(layers, block)
        push!(layers, Lux.WrappedFunction(relu))
    end

    # Output head
    push!(layers, Dense(hidden => d_out))

    backbone = Chain(layers...)

    return backbone, embed_dim
end

# ═══════════════════════════════════════════════════════════════════════════
# 5. Forward Diffusion
# ═══════════════════════════════════════════════════════════════════════════

"""
Gaussian forward diffusion: add noise to numeric features at timestep t.

    x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
"""
function _gaussian_forward(x0, alphas_cumprod, t, rng)
    batch = size(x0, 2)
    abar  = alphas_cumprod[t]'                  # (1, batch)
    sqrt_abar       = sqrt.(abar)
    sqrt_one_m_abar = sqrt.(1f0 .- abar)
    ε = randn(rng, Float32, size(x0)...)
    x_t = sqrt_abar .* x0 .+ sqrt_one_m_abar .* ε
    return x_t, ε
end

"""
Multinomial forward diffusion: corrupt categorical one-hot at timestep t.

    q(x_t | x_0) = Cat(ᾱ_t · x_0 + (1 - ᾱ_t) / K)

Returns softened one-hot vectors (not re-sampled — used directly as input).
"""
function _multinomial_forward(x0_oh, cat_dims, alphas_cumprod, t)
    batch = size(x0_oh, 2)
    abar  = alphas_cumprod[t]'                  # (1, batch)

    x_t = similar(x0_oh)
    offset = 0
    for K in cat_dims
        block = @view x0_oh[(offset + 1):(offset + K), :]
        uniform = 1f0 / K
        x_t[(offset + 1):(offset + K), :] = abar .* block .+ (1f0 .- abar) .* uniform
        offset += K
    end
    return x_t
end

# ═══════════════════════════════════════════════════════════════════════════
# 6. Loss Function
# ═══════════════════════════════════════════════════════════════════════════

"""
Combined loss: MSE on Gaussian noise prediction + cross-entropy on
multinomial logits.
"""
function _diffusion_loss(backbone, emb_layer, ps_backbone, ps_emb,
                         st_backbone, st_emb,
                         X_num_noised, X_cat_noised, cat_indices_batch,
                         t_batch, ε_true, d_num, cat_dims)
    # Timestep embedding
    t_emb, st_emb_new = Lux.apply(emb_layer, t_batch, ps_emb, st_emb)

    # Concatenate input (no mutation — Zygote-safe)
    input = if d_num > 0 && length(cat_dims) > 0
        vcat(X_num_noised, X_cat_noised, t_emb)
    elseif d_num > 0
        vcat(X_num_noised, t_emb)
    elseif length(cat_dims) > 0
        vcat(X_cat_noised, t_emb)
    else
        t_emb
    end

    # Forward pass
    output, st_bb_new = Lux.apply(backbone, input, ps_backbone, st_backbone)

    loss = 0f0
    batch = size(output, 2)

    # ── Gaussian MSE loss ──────────────────────────────────────────────
    if d_num > 0
        ε_pred = output[1:d_num, :]
        loss += sum(abs2, ε_pred .- ε_true) / (d_num * batch)
    end

    # ── Multinomial cross-entropy loss ─────────────────────────────────
    offset = d_num
    n_cat  = length(cat_dims)
    if n_cat > 0
        ce = 0f0
        for (c, K) in enumerate(cat_dims)
            logits = output[(offset + 1):(offset + K), :]   # (K, batch)
            # Numerically stable log-softmax
            m = maximum(logits; dims = 1)
            log_probs = logits .- m .- log.(sum(exp.(logits .- m); dims = 1))
            # Gather the log-prob of the true class
            for i in 1:batch
                k = cat_indices_batch[c, i]
                ce -= log_probs[k, i]
            end
            offset += K
        end
        loss += ce / (n_cat * batch)
    end

    return loss, (st_bb_new, st_emb_new)
end

# ═══════════════════════════════════════════════════════════════════════════
# 7. Gradient tree utilities
# ═══════════════════════════════════════════════════════════════════════════

"""Compute the squared L2 norm of a gradient tree (NamedTuple of arrays)."""
function _grad_sqnorm(gs)
    s = 0.0
    _grad_sqnorm_accum!(s, gs)
end

function _grad_sqnorm_accum!(s, gs::NamedTuple)
    for v in values(gs)
        s = _grad_sqnorm_accum!(s, v)
    end
    return s
end

function _grad_sqnorm_accum!(s, gs::AbstractArray)
    return s + sum(abs2, gs)
end

_grad_sqnorm_accum!(s, ::Nothing) = s
_grad_sqnorm_accum!(s, ::Tuple{}) = s

function _grad_sqnorm_accum!(s, gs::Tuple)
    for v in gs
        s = _grad_sqnorm_accum!(s, v)
    end
    return s
end

"""Scale a gradient tree by a scalar factor."""
function _grad_scale(gs::NamedTuple, α)
    return NamedTuple{keys(gs)}(map(v -> _grad_scale(v, α), values(gs)))
end
_grad_scale(gs::AbstractArray, α) = gs .* Float32(α)
_grad_scale(::Nothing, _) = nothing
_grad_scale(gs::Tuple, α) = map(v -> _grad_scale(v, α), gs)

"""Add two gradient trees element-wise."""
function _grad_add(a::NamedTuple, b::NamedTuple)
    return NamedTuple{keys(a)}(map((va, vb) -> _grad_add(va, vb), values(a), values(b)))
end
_grad_add(a::AbstractArray, b::AbstractArray) = a .+ b
_grad_add(::Nothing, ::Nothing) = nothing
_grad_add(a, ::Nothing) = a         # Zygote returns nothing for param-free layers
_grad_add(::Nothing, b) = b
_grad_add(a::Tuple, b::Tuple) = map(_grad_add, a, b)

"""Zero-initialize a gradient tree matching the structure of `ps`."""
function _grad_zero(ps::NamedTuple)
    return NamedTuple{keys(ps)}(map(_grad_zero, values(ps)))
end
_grad_zero(ps::AbstractArray) = zero(ps)
_grad_zero(::Nothing) = nothing
_grad_zero(ps::Tuple) = map(_grad_zero, ps)

"""Add Gaussian noise to a gradient tree."""
function _grad_add_noise!(gs::NamedTuple, σ::Float64, rng)
    return NamedTuple{keys(gs)}(map(v -> _grad_add_noise!(v, σ, rng), values(gs)))
end
function _grad_add_noise!(gs::AbstractArray, σ::Float64, rng)
    gs .+= randn(rng, Float32, size(gs)...) .* Float32(σ)
    return gs
end
_grad_add_noise!(::Nothing, _, _) = nothing
_grad_add_noise!(gs::Tuple, σ, rng) = map(v -> _grad_add_noise!(v, σ, rng), gs)

# ═══════════════════════════════════════════════════════════════════════════
# 8. AD Backend Abstraction (REQ-DIF-009)
# ═══════════════════════════════════════════════════════════════════════════

# The single dispatch point for computing gradients.  Both _train_standard!
# and _train_dpsgd! route through this function.  Adding Enzyme support
# means adding one method:
#
#   function _compute_grad(::Val{:enzyme}, loss_fn, ps) ... end
#
const AD_BACKEND = Val(:zygote)

"""
    _compute_grad(loss_fn, backend, ps) -> (loss, aux, grad)

Compute the gradient of `loss_fn(ps)` with respect to `ps`.
`loss_fn` must return `(scalar_loss, aux)`.
Returns `(loss, aux, grad)` where `grad` mirrors the structure of `ps`.

The `do`-block convention places the closure first, so call sites read:

    _compute_grad(AD_BACKEND, ps) do p
        my_loss(p)
    end
"""
function _compute_grad(loss_fn, ::Val{:zygote}, ps)
    (loss, aux), gs = Zygote.withgradient(loss_fn, ps)
    return loss, aux, gs[1]
end

# ═══════════════════════════════════════════════════════════════════════════
# 9. Standard Training Loop
# ═══════════════════════════════════════════════════════════════════════════

function _train_standard!(backbone, emb_layer, ps_bb, ps_emb,
                          st_bb, st_emb,
                          X_num, X_cat_oh, cat_indices,
                          betas, alphas_cumprod, cat_dims, d_num,
                          epochs, batch_size, rng)
    T     = length(betas)
    nrows = size(X_num, 2) > 0 ? size(X_num, 2) : size(X_cat_oh, 2)

    # Merge params for optimizer
    ps_all    = (; backbone = ps_bb, emb = ps_emb)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps_all)

    for epoch in 1:epochs
        perm = Random.randperm(rng, nrows)
        for start in 1:batch_size:nrows
            stop = min(start + batch_size - 1, nrows)
            idx  = perm[start:stop]
            bs   = length(idx)

            t_batch = rand(rng, 1:T, bs)

            # Forward diffusion
            x_num_batch = d_num > 0 ? X_num[:, idx] : zeros(Float32, 0, bs)
            x_cat_batch = size(X_cat_oh, 1) > 0 ? X_cat_oh[:, idx] : zeros(Float32, 0, bs)
            cat_idx_batch = size(cat_indices, 1) > 0 ? cat_indices[:, idx] : zeros(Int, 0, bs)

            if d_num > 0
                x_num_noised, ε = _gaussian_forward(x_num_batch, alphas_cumprod, t_batch, rng)
            else
                x_num_noised = zeros(Float32, 0, bs)
                ε = zeros(Float32, 0, bs)
            end

            if size(x_cat_batch, 1) > 0
                x_cat_noised = _multinomial_forward(x_cat_batch, cat_dims, alphas_cumprod, t_batch)
            else
                x_cat_noised = zeros(Float32, 0, bs)
            end

            # Gradient (dispatched through AD backend — REQ-DIF-009)
            loss, states_new, g = _compute_grad(AD_BACKEND, ps_all) do p
                _diffusion_loss(backbone, emb_layer,
                                p.backbone, p.emb,
                                st_bb, st_emb,
                                x_num_noised, x_cat_noised, cat_idx_batch,
                                t_batch, ε, d_num, cat_dims)
            end
            st_bb, st_emb = states_new

            opt_state, ps_all = Optimisers.update(opt_state, ps_all, g)
        end
    end

    return ps_all.backbone, ps_all.emb, st_bb, st_emb
end

# ═══════════════════════════════════════════════════════════════════════════
# 10. DP-SGD Training Loop
# ═══════════════════════════════════════════════════════════════════════════

"""
Rényi DP accountant: compute (ε, δ)-DP spent after `steps` applications of
the Gaussian mechanism with noise multiplier `σ` and sampling rate `q`.

Uses the Rényi divergence bound [Mironov 2017]:
    ε(α) = (1/(α-1)) log(E[exp((α-1) · privacy_loss)])

For the subsampled Gaussian mechanism, we use the standard bound:
    ε_RDP(α) ≤ (1/(α-1)) log(1 + q²α(α-1) / (2σ²))
summed over `steps` compositions.  Convert to (ε, δ)-DP via:
    ε = min_α { ε_RDP(α) + log(1/δ) / (α-1) }
"""
function _rdp_accountant(σ::Float64, q::Float64, steps::Int, delta::Float64)
    alphas = vcat(collect(2:10), collect(12:2:64), [128, 256])
    best_eps = Inf

    for α in alphas
        # RDP per step (subsampled Gaussian mechanism)
        log_term = log(1 + q^2 * α * (α - 1) / (2 * σ^2))
        rdp = log_term / (α - 1)
        rdp_total = rdp * steps

        # Convert RDP → (ε, δ)-DP
        eps = rdp_total + log(1 / delta) / (α - 1)
        best_eps = min(best_eps, eps)
    end

    return best_eps
end

function _train_dpsgd!(backbone, emb_layer, ps_bb, ps_emb,
                       st_bb, st_emb,
                       X_num, X_cat_oh, cat_indices,
                       betas, alphas_cumprod, cat_dims, d_num,
                       epochs, batch_size, privacy, rng)
    T     = length(betas)
    nrows = size(X_num, 2) > 0 ? size(X_num, 2) : size(X_cat_oh, 2)

    # DP-SGD parameters
    C = 1.0                                     # gradient clip norm
    q = min(batch_size / nrows, 1.0)            # sampling rate
    total_steps = epochs * ceil(Int, nrows / batch_size)

    # Binary search for noise multiplier σ that satisfies the budget
    σ_lo, σ_hi = 0.1, 100.0
    for _ in 1:64
        σ_mid = (σ_lo + σ_hi) / 2
        eps = _rdp_accountant(σ_mid, q, total_steps, privacy.delta)
        if eps > privacy.epsilon
            σ_lo = σ_mid
        else
            σ_hi = σ_mid
        end
    end
    σ_noise = σ_hi
    @info "DP-SGD: σ_noise = $(round(σ_noise; digits=3)), " *
          "total_steps = $total_steps, " *
          "achieved ε ≈ $(round(_rdp_accountant(σ_noise, q, total_steps, privacy.delta); digits=3))"

    ps_all    = (; backbone = ps_bb, emb = ps_emb)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps_all)

    for epoch in 1:epochs
        perm = Random.randperm(rng, nrows)
        for start in 1:batch_size:nrows
            stop = min(start + batch_size - 1, nrows)
            idx  = perm[start:stop]
            bs   = length(idx)

            t_batch = rand(rng, 1:T, bs)

            x_num_batch = d_num > 0 ? X_num[:, idx] : zeros(Float32, 0, bs)
            x_cat_batch = size(X_cat_oh, 1) > 0 ? X_cat_oh[:, idx] : zeros(Float32, 0, bs)
            cat_idx_batch = size(cat_indices, 1) > 0 ? cat_indices[:, idx] : zeros(Int, 0, bs)

            if d_num > 0
                x_num_noised, ε = _gaussian_forward(x_num_batch, alphas_cumprod, t_batch, rng)
            else
                x_num_noised = zeros(Float32, 0, bs)
                ε = zeros(Float32, 0, bs)
            end

            if size(x_cat_batch, 1) > 0
                x_cat_noised = _multinomial_forward(x_cat_batch, cat_dims, alphas_cumprod, t_batch)
            else
                x_cat_noised = zeros(Float32, 0, bs)
            end

            # ── Per-sample gradient clipping ────────────────────────────
            gs_sum = _grad_zero(ps_all)

            for si in 1:bs
                xn_i  = d_num > 0 ? x_num_noised[:, si:si] : zeros(Float32, 0, 1)
                xc_i  = size(x_cat_noised, 1) > 0 ? x_cat_noised[:, si:si] : zeros(Float32, 0, 1)
                ci_i  = size(cat_idx_batch, 1) > 0 ? cat_idx_batch[:, si:si] : zeros(Int, 0, 1)
                t_i   = [t_batch[si]]
                ε_i   = d_num > 0 ? ε[:, si:si] : zeros(Float32, 0, 1)

                _, _, g = _compute_grad(AD_BACKEND, ps_all) do p
                    _diffusion_loss(backbone, emb_layer,
                                    p.backbone, p.emb,
                                    st_bb, st_emb,
                                    xn_i, xc_i, ci_i,
                                    t_i, ε_i, d_num, cat_dims)
                end
                gnorm = sqrt(_grad_sqnorm(g))
                clip_factor = min(1.0, C / max(gnorm, 1e-12))
                g_clipped = _grad_scale(g, clip_factor)
                gs_sum = _grad_add(gs_sum, g_clipped)
            end

            # Average and add noise
            gs_avg = _grad_scale(gs_sum, 1.0 / bs)
            noise_scale = σ_noise * C / bs
            gs_noisy = _grad_add_noise!(gs_avg, noise_scale, rng)

            opt_state, ps_all = Optimisers.update(opt_state, ps_all, gs_noisy)
        end
    end

    return ps_all.backbone, ps_all.emb, st_bb, st_emb
end

# ═══════════════════════════════════════════════════════════════════════════
# 11. Reverse Sampling (Denoising)
# ═══════════════════════════════════════════════════════════════════════════

"""
Sample `n` rows from a trained TabDDPM model via reverse denoising.
"""
function _denoise_sample(backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
                         betas, alphas_cumprod, d_num, cat_dims, n, rng)
    T = length(betas)
    d_cat_total = sum(cat_dims; init = 0)
    alphas = 1f0 .- betas

    # Initialize from pure noise
    x_num = d_num > 0 ? randn(rng, Float32, d_num, n) : zeros(Float32, 0, n)
    x_cat = d_cat_total > 0 ? begin
        # Uniform initialization for categoricals
        oh = zeros(Float32, d_cat_total, n)
        offset = 0
        for K in cat_dims
            oh[(offset + 1):(offset + K), :] .= 1f0 / K
            offset += K
        end
        oh
    end : zeros(Float32, 0, n)

    for t in T:-1:1
        t_batch = fill(t, n)

        # Timestep embedding
        t_emb, _ = Lux.apply(emb_layer, t_batch, ps_emb, st_emb)

        # Concatenate
        input = if d_num > 0 && d_cat_total > 0
            vcat(x_num, x_cat, t_emb)
        elseif d_num > 0
            vcat(x_num, t_emb)
        elseif d_cat_total > 0
            vcat(x_cat, t_emb)
        else
            t_emb
        end

        output, _ = Lux.apply(backbone, input, ps_bb, st_bb)

        # ── Gaussian reverse step ──────────────────────────────────────
        if d_num > 0
            ε_pred = output[1:d_num, :]
            α_t    = alphas[t]
            ᾱ_t    = alphas_cumprod[t]
            β_t    = betas[t]

            # DDPM reverse mean: μ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_pred)
            coef1 = 1f0 / sqrt(α_t)
            coef2 = β_t / sqrt(1f0 - ᾱ_t + 1f-8)
            mean  = coef1 .* (x_num .- coef2 .* ε_pred)

            if t > 1
                σ_t = sqrt(β_t)
                z   = randn(rng, Float32, d_num, n)
                x_num = mean .+ σ_t .* z
            else
                x_num = mean
            end
        end

        # ── Multinomial reverse step ───────────────────────────────────
        if d_cat_total > 0
            offset_in  = d_num
            offset_out = 0
            for K in cat_dims
                logits = output[(offset_in + 1):(offset_in + K), :]

                # Predict x_0 probabilities from logits (softmax)
                m = maximum(logits; dims = 1)
                exp_logits = exp.(logits .- m)
                probs_x0 = exp_logits ./ sum(exp_logits; dims = 1)

                if t > 1
                    # Posterior: q(x_{t-1} | x_t, x_0_pred)
                    # Use the predicted x_0 to compute the posterior distribution
                    ᾱ_t   = alphas_cumprod[t]
                    ᾱ_tm1 = t > 1 ? alphas_cumprod[t - 1] : 1f0
                    uniform = 1f0 / K

                    # Compute posterior for each sample
                    for i in 1:n
                        p0 = probs_x0[:, i]
                        # q(x_{t-1} | x_0) for each possible x_0 category
                        # ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
                        # Simplified: use predicted x_0 probs directly, re-noise to t-1
                        posterior = ᾱ_tm1 .* p0 .+ (1f0 - ᾱ_tm1) * uniform
                        posterior ./= sum(posterior)
                        # Sample from posterior
                        k = StatsBase.sample(rng, 1:K, StatsBase.Weights(max.(posterior, 0f0)))
                        x_cat[(offset_out + 1):(offset_out + K), i] .= 0f0
                        x_cat[offset_out + k, i] = 1f0
                    end
                else
                    # Final step: argmax
                    for i in 1:n
                        k = argmax(probs_x0[:, i])
                        x_cat[(offset_out + 1):(offset_out + K), i] .= 0f0
                        x_cat[offset_out + k, i] = 1f0
                    end
                end

                offset_in  += K
                offset_out += K
            end
        end
    end

    return x_num, x_cat
end

# ═══════════════════════════════════════════════════════════════════════════
# 12. _fit_engine(::DiffusionGenerator, …)
# ═══════════════════════════════════════════════════════════════════════════

function _fit_engine(gen::DiffusionGenerator, cols, col_names, id_set, fill_dict,
                     hints, nm_cache, basetype_cache, nrows, mat, rng, privacy)
    hint_dict = Dict(h.name => h for h in hints)
    col_kinds = Symbol[]
    miss      = Dict{Symbol, Float64}()

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
            hint.kind
        else
            _detect_column_type(nm, T)
        end
        push!(col_kinds, kind)
    end

    # ── Preprocess ─────────────────────────────────────────────────────
    X_num, X_cat_oh, cat_indices, info = _preprocess(
        cols, col_names, col_kinds, id_set, hints, nrows)

    d_num       = info.d_num
    d_cat_total = info.d_cat_total
    cat_dims_v  = info.cat_dims

    if d_num == 0 && d_cat_total == 0
        error("No statistical columns for DiffusionGenerator after preprocessing.")
    end

    # ── Noise schedule ─────────────────────────────────────────────────
    n_steps = 1000
    betas, alphas_cumprod = _linear_schedule(n_steps)

    # ── Build model ────────────────────────────────────────────────────
    embed_dim = 128
    hidden    = min(256, max(64, 4 * (d_num + d_cat_total)))
    n_blocks  = 4
    d_in      = d_num + d_cat_total + embed_dim

    backbone, _ = _build_model(d_in, d_num, cat_dims_v;
                               hidden = hidden, n_blocks = n_blocks,
                               embed_dim = embed_dim, dropout = 0.0)
    emb_layer = SinusoidalEmbedding(embed_dim)

    lux_rng = Random.MersenneTwister(42)  # deterministic init
    ps_bb, st_bb   = Lux.setup(lux_rng, backbone)
    ps_emb, st_emb = Lux.setup(lux_rng, emb_layer)

    # ── Train ──────────────────────────────────────────────────────────
    if gen.dp
        ps_bb, ps_emb, st_bb, st_emb = _train_dpsgd!(
            backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
            X_num, X_cat_oh, cat_indices,
            betas, alphas_cumprod, cat_dims_v, d_num,
            gen.epochs, gen.batch_size, privacy, rng)
    else
        ps_bb, ps_emb, st_bb, st_emb = _train_standard!(
            backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
            X_num, X_cat_oh, cat_indices,
            betas, alphas_cumprod, cat_dims_v, d_num,
            gen.epochs, gen.batch_size, rng)
    end

    id_cols = [name for name in col_names if name in id_set]

    # Store both model components together
    full_model = (; backbone = backbone, emb = emb_layer)
    full_ps    = (; backbone = ps_bb, emb = ps_emb)
    full_st    = (; backbone = st_bb, emb = st_emb)

    return FittedDiffusionModel(
        col_names, col_kinds,
        info.num_cols, info.cat_cols,
        info.num_means, info.num_stds,
        info.cat_levels, info.cat_dims,
        full_model, full_ps, full_st,
        n_steps, betas, alphas_cumprod,
        miss, nrows,
        id_cols, fill_dict, mat, rng,
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# 13. sample(::FittedDiffusionModel, …)
# ═══════════════════════════════════════════════════════════════════════════

function sample(model::FittedDiffusionModel, n::Int;
                rng::AbstractRNG = model.rng)
    n ≥ 1 || throw(ArgumentError("n must be at least 1, got $n"))

    if n > 10 * model.n_original
        @warn "Requested n ($n) is more than 10× the original " *
              "($(model.n_original) rows)."
    end

    full_model = model.lux_model
    full_ps    = model.trained_params
    full_st    = model.model_state

    backbone  = full_model.backbone
    emb_layer = full_model.emb
    ps_bb     = full_ps.backbone
    ps_emb    = full_ps.emb
    st_bb     = full_st.backbone
    st_emb    = full_st.emb

    d_num     = length(model.num_columns)
    cat_dims  = model.cat_dims

    # ── Reverse denoise ────────────────────────────────────────────────
    x_num, x_cat = _denoise_sample(
        backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
        model.betas, model.alphas_cumprod,
        d_num, cat_dims, n, rng)

    # ── Unpack into result dict ────────────────────────────────────────
    result = Dict{Symbol, Vector}()

    # Numeric columns: reverse z-score normalization
    for (j, name) in enumerate(model.num_columns)
        μ = model.num_means[j]
        σ = model.num_stds[j]
        raw = x_num[j, :] .* σ .+ μ

        # Find original kind
        idx = findfirst(==(name), model.column_names)
        kind = model.column_kinds[idx]
        if kind == :integer
            result[name] = round.(Int64, raw)
        else
            result[name] = Float64.(raw)
        end
    end

    # Categorical columns: decode one-hot
    offset = 0
    for (c, name) in enumerate(model.cat_columns)
        K = cat_dims[c]
        lvls = model.cat_levels[name]
        vals = Vector{eltype(lvls)}(undef, n)
        for i in 1:n
            k = argmax(x_cat[(offset + 1):(offset + K), i])
            vals[i] = lvls[k]
        end
        result[name] = vals
        offset += K
    end

    # Constant columns: not modelled by the network, fill with missing
    # (the original constant value is unknown to the diffusion model;
    #  _postprocess handles missingness injection)
    for (i, name) in enumerate(model.column_names)
        kind = model.column_kinds[i]
        kind == :identifier && continue
        kind == :constant || continue
        haskey(result, name) && continue
        result[name] = fill(missing, n)
    end

    return _postprocess(result, model, n, rng)
end

end # module DataMimicLuxExt
