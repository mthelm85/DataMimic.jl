# ─── DataMimicLuxExt ────────────────────────────────────────────────────────
#
# Package extension: DiffusionGenerator engine (TabDDPM).
#
# Loaded when both Lux.jl and Zygote.jl are present.
# Implements REQ-DIF-001 through REQ-DIF-009.
#
# References:
#   [Kotelnikov et al. 2023]   — TabDDPM architecture (ResNet MLP + additive
#                                 timestep conditioning + LayerNorm + SiLU)
#   [Ho et al. 2020]           — DDPM (Gaussian diffusion)
#   [Song et al. 2020]         — DDIM (deterministic reverse sampling)
#   [Hoogeboom et al. 2021]    — Multinomial diffusion
#   [Abadi et al. 2016]        — DP-SGD
#   [Mironov et al. 2019]      — Exact subsampled Gaussian RDP accounting

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
import LinearAlgebra

# Custom relu that bypasses cuDNN (avoids CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
# when cuDNN is missing or version-mismatched).  Pure CUDA element-wise kernel.
_fast_relu(x) = max(x, zero(x))

# SiLU / Swish activation — used in timestep embedding MLP and ResNet blocks
# [Kotelnikov et al. 2023, TabDDPM architecture].
_fast_silu(x) = x / (1f0 + exp(-x))

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
    # t: (batch,) — Float32 timesteps, expected on same device as st.freqs.
    # Callers must move t to device before calling.
    t_flat = vec(t)                 # (batch,) on device
    freqs  = st.freqs               # (half,) on device
    args   = t_flat' .* freqs       # (half, batch)
    emb    = vcat(sin.(args), cos.(args))  # (dim, batch)
    return emb, st
end

Lux.statelength(l::SinusoidalEmbedding) = l.dim ÷ 2

# ═══════════════════════════════════════════════════════════════════════════
# 4. TabDDPM Backbone  [Kotelnikov et al. 2023]
# ═══════════════════════════════════════════════════════════════════════════
#
# Architecture (following the TabDDPM ResNet MLP):
#   1. Sinusoidal timestep embedding → 2-layer MLP with SiLU
#   2. Input features → Dense projection → + time embedding
#   3. ResNet blocks: [LayerNorm → SiLU → Dense → SiLU → (Dropout) → Dense]
#      with residual skip connections
#   4. Output head:  LayerNorm → SiLU → Dense → d_out

"""
Lux container layer for the TabDDPM denoising network.

Subcomponents:
- `proj`:     Dense(d_in → hidden)  — input feature projection
- `time_mlp`: Chain(Dense → SiLU → Dense) — timestep embedding projection
- `blocks`:   Chain of SkipConnection ResNet blocks with LayerNorm + SiLU
- `out_head`: Chain(LayerNorm → SiLU → Dense) — output projection

Call signature: `(model)((features, t_emb), ps, st)` where
- `features`: (d_in, batch) — concatenated numeric + categorical features
- `t_emb`:    (embed_dim, batch) — raw sinusoidal timestep embedding
"""
struct TabDDPMBackbone <: Lux.AbstractLuxContainerLayer{(:proj, :time_mlp, :blocks, :out_head)}
    proj
    time_mlp
    blocks
    out_head
end

function (m::TabDDPMBackbone)((features, t_emb), ps, st)
    # Project input features: (d_in, B) → (hidden, B)
    h, st_proj = Lux.apply(m.proj, features, ps.proj, st.proj)

    # Project timestep embedding: (embed_dim, B) → (hidden, B), then add
    t_proj, st_tmlp = Lux.apply(m.time_mlp, t_emb, ps.time_mlp, st.time_mlp)
    h = h .+ t_proj

    # ResNet blocks (residual connections handled by SkipConnection)
    h, st_blocks = Lux.apply(m.blocks, h, ps.blocks, st.blocks)

    # Output head
    output, st_head = Lux.apply(m.out_head, h, ps.out_head, st.out_head)

    st_new = (; proj = st_proj, time_mlp = st_tmlp,
                blocks = st_blocks, out_head = st_head)
    return output, st_new
end

"""
Build the TabDDPM denoising network.

Architecture: proj(features) + time_mlp(t_emb) → ResNet blocks → output.

- `d_in`:     feature dimension (d_num + d_cat_onehot) — no timestep embedding
- `d_num`:    number of numeric output channels (predict noise ε)
- `cat_dims`: per-categorical output logits
- `hidden`:   hidden dimension
- `n_blocks`: number of residual blocks
- `embed_dim`:timestep embedding dimension
"""
function _build_model(d_in::Int, d_num::Int, cat_dims::Vector{Int};
                      hidden::Int = 256, n_blocks::Int = 4,
                      embed_dim::Int = 128, dropout::Float64 = 0.0)
    d_out = d_num + sum(cat_dims; init = 0)

    # Input projection (no activation — first block has pre-activation norm)
    proj = Dense(d_in => hidden)

    # Time embedding MLP: embed_dim → hidden with SiLU
    time_mlp = Chain(
        Dense(embed_dim => hidden, _fast_silu),
        Dense(hidden => hidden))

    # Residual blocks: pre-activation ResNet [LayerNorm → SiLU → Dense → SiLU → Dense]
    block_layers = []
    for _ in 1:n_blocks
        inner = if dropout > 0
            Chain(
                Lux.LayerNorm((hidden,)),
                Lux.WrappedFunction(x -> _fast_silu.(x)),
                Dense(hidden => hidden, _fast_silu),
                Lux.Dropout(Float32(dropout)),
                Dense(hidden => hidden))
        else
            Chain(
                Lux.LayerNorm((hidden,)),
                Lux.WrappedFunction(x -> _fast_silu.(x)),
                Dense(hidden => hidden, _fast_silu),
                Dense(hidden => hidden))
        end
        push!(block_layers, SkipConnection(inner, +))
    end
    blocks = Chain(block_layers...)

    # Output head: LayerNorm → SiLU → Dense
    out_head = Chain(
        Lux.LayerNorm((hidden,)),
        Lux.WrappedFunction(x -> _fast_silu.(x)),
        Dense(hidden => d_out))

    model = TabDDPMBackbone(proj, time_mlp, blocks, out_head)
    return model, embed_dim
end

# ═══════════════════════════════════════════════════════════════════════════
# 5. Forward Diffusion
# ═══════════════════════════════════════════════════════════════════════════

"""
Gaussian forward diffusion: add noise to numeric features at timestep t.

    x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
"""
function _gaussian_forward(x0, alphas_cumprod, t, rng, dev)
    batch = size(x0, 2)
    # Index alphas_cumprod on CPU to avoid GPU scalar indexing
    abar_cpu = Float32.(alphas_cumprod[t])'      # (1, batch) on CPU
    abar     = _to_device(abar_cpu, dev)
    sqrt_abar       = sqrt.(abar)
    sqrt_one_m_abar = sqrt.(1f0 .- abar)
    ε = _to_device(randn(rng, Float32, size(x0)...), dev)
    x_t = sqrt_abar .* x0 .+ sqrt_one_m_abar .* ε
    return x_t, ε
end

"""
Multinomial forward diffusion: corrupt categorical one-hot at timestep t.

    q(x_t | x_0) = Cat(ᾱ_t · x_0 + (1 - ᾱ_t) / K)

Returns softened one-hot vectors (not re-sampled — used directly as input).
"""
function _multinomial_forward(x0_oh, cat_dims, alphas_cumprod, t, dev)
    batch = size(x0_oh, 2)
    # Index alphas_cumprod on CPU to avoid GPU scalar indexing
    abar_cpu = Float32.(alphas_cumprod[t])'      # (1, batch) on CPU
    abar     = _to_device(abar_cpu, dev)

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

`x_cat_original` is the un-noised one-hot encoding of the true
categorical labels (same layout as `X_cat_noised`).  Using it avoids
scalar indexing into GPU arrays.
"""
function _diffusion_loss(backbone, emb_layer, ps_backbone, ps_emb,
                         st_backbone, st_emb,
                         X_num_noised, X_cat_noised, x_cat_original,
                         t_batch, ε_true, d_num, cat_dims)
    # Timestep embedding (sinusoidal)
    t_emb, st_emb_new = Lux.apply(emb_layer, t_batch, ps_emb, st_emb)

    # Feature input (timestep conditioning handled inside backbone via addition)
    features = if d_num > 0 && length(cat_dims) > 0
        vcat(X_num_noised, X_cat_noised)
    elseif d_num > 0
        X_num_noised
    else
        X_cat_noised
    end

    # Forward pass (backbone projects features + t_emb, adds, then ResNet)
    output, st_bb_new = Lux.apply(backbone, (features, t_emb), ps_backbone, st_backbone)

    loss = 0f0
    batch = size(output, 2)

    # ── Gaussian MSE loss ──────────────────────────────────────────────
    if d_num > 0
        ε_pred = output[1:d_num, :]
        loss += sum(abs2, ε_pred .- ε_true) / (d_num * batch)
    end

    # ── Multinomial cross-entropy loss (GPU-safe, no scalar indexing) ──
    offset = d_num
    cat_offset = 0
    n_cat  = length(cat_dims)
    if n_cat > 0
        ce = 0f0
        for K in cat_dims
            logits = output[(offset + 1):(offset + K), :]   # (K, batch)
            # Numerically stable log-softmax
            m = maximum(logits; dims = 1)
            log_probs = logits .- m .- log.(sum(exp.(logits .- m); dims = 1))
            # Vectorized CE: one-hot target · log_probs
            oh_target = x_cat_original[(cat_offset + 1):(cat_offset + K), :]
            ce -= sum(oh_target .* log_probs)
            offset += K
            cat_offset += K
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
    noise = randn(rng, Float32, size(gs)...)
    # Match device of gradient array (GPU-safe)
    noise_d = similar(gs)
    copyto!(noise_d, noise)
    gs .+= noise_d .* Float32(σ)
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
# 8b. Device Abstraction (REQ-DIF-010 through REQ-DIF-013)
# ═══════════════════════════════════════════════════════════════════════════

# Auto-detect GPU if LuxCUDA / Metal / AMDGPU is loaded; otherwise CPU.
# No GPU package is a dependency — the user opts in by loading LuxCUDA.

"""
    _get_devices() -> (compute_device, cpu_device)

Return the best available compute device and the CPU device.
If CUDA/Metal/ROCm is loaded, returns the GPU device; otherwise CPU.
"""
function _get_devices()
    gdev = Lux.gpu_device(; force = false)  # returns CPU if no GPU
    cdev = Lux.cpu_device()
    return gdev, cdev
end

"""
    _to_device(x, dev)

Move `x` to the given device. For arrays, NamedTuples, and Lux states.
"""
_to_device(x, dev) = dev(x)

# ═══════════════════════════════════════════════════════════════════════════
# 9. Standard Training Loop
# ═══════════════════════════════════════════════════════════════════════════

function _train_standard!(backbone, emb_layer, ps_bb, ps_emb,
                          st_bb, st_emb,
                          X_num, X_cat_oh, cat_indices,
                          betas, alphas_cumprod, cat_dims, d_num,
                          epochs, batch_size, rng, dev)
    T     = length(betas)
    nrows = size(X_num, 2) > 0 ? size(X_num, 2) : size(X_cat_oh, 2)

    # Move training data and params to device (GPU if available)
    X_num_d     = _to_device(X_num, dev)
    X_cat_oh_d  = _to_device(X_cat_oh, dev)
    cat_indices_d = _to_device(cat_indices, dev)
    alphas_cumprod_d = _to_device(alphas_cumprod, dev)

    # Merge params for optimizer
    ps_all    = _to_device((; backbone = ps_bb, emb = ps_emb), dev)
    st_bb     = _to_device(st_bb, dev)
    st_emb    = _to_device(st_emb, dev)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps_all)

    t_start    = time()
    epoch_loss = 0.0
    n_batches  = 0

    # Progress reporting interval: ~20 updates over the full run
    report_every = max(1, epochs ÷ 20)

    for epoch in 1:epochs
        epoch_loss = 0.0
        n_batches  = 0
        perm = Random.randperm(rng, nrows)
        for start in 1:batch_size:nrows
            stop = min(start + batch_size - 1, nrows)
            idx  = perm[start:stop]
            bs   = length(idx)

            t_batch = rand(rng, 1:T, bs)
            # Move timesteps to device as Float32 (outside AD-traced code)
            t_batch_d = _to_device(Float32.(t_batch), dev)

            # Forward diffusion (slice on device)
            x_num_batch = d_num > 0 ? X_num_d[:, idx] : _to_device(zeros(Float32, 0, bs), dev)
            x_cat_batch = size(X_cat_oh_d, 1) > 0 ? X_cat_oh_d[:, idx] : _to_device(zeros(Float32, 0, bs), dev)

            if d_num > 0
                x_num_noised, ε = _gaussian_forward(x_num_batch, alphas_cumprod_d, t_batch, rng, dev)
            else
                x_num_noised = _to_device(zeros(Float32, 0, bs), dev)
                ε = _to_device(zeros(Float32, 0, bs), dev)
            end

            if size(x_cat_batch, 1) > 0
                x_cat_noised = _multinomial_forward(x_cat_batch, cat_dims, alphas_cumprod_d, t_batch, dev)
            else
                x_cat_noised = _to_device(zeros(Float32, 0, bs), dev)
            end

            # Gradient (dispatched through AD backend — REQ-DIF-009)
            loss, states_new, g = _compute_grad(AD_BACKEND, ps_all) do p
                _diffusion_loss(backbone, emb_layer,
                                p.backbone, p.emb,
                                st_bb, st_emb,
                                x_num_noised, x_cat_noised, x_cat_batch,
                                t_batch_d, ε, d_num, cat_dims)
            end
            st_bb, st_emb = states_new
            epoch_loss += loss
            n_batches  += 1

            opt_state, ps_all = Optimisers.update(opt_state, ps_all, g)
        end

        # Progress report
        if epoch == 1 || epoch % report_every == 0 || epoch == epochs
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed  = time() - t_start
            eta      = elapsed / epoch * (epochs - epoch)
            @info "Epoch $(epoch)/$(epochs)  loss=$(round(avg_loss; digits=4))  elapsed=$(round(Int, elapsed))s  ETA=$(round(Int, eta))s"
        end
    end

    return ps_all.backbone, ps_all.emb, st_bb, st_emb
end

# ═══════════════════════════════════════════════════════════════════════════
# 10. DP-SGD Training Loop
# ═══════════════════════════════════════════════════════════════════════════

"""Log-sum-exp with numerical stability."""
function _logsumexp(xs)
    m = maximum(xs)
    isinf(m) && return m
    return m + log(sum(exp.(xs .- m)))
end

"""Log of n! — hand-rolled to avoid a SpecialFunctions.jl dependency."""
function _logfactorial(n::Int)
    n <= 1 && return 0.0
    return sum(log(Float64(i)) for i in 2:n)
end

"""
Rényi DP accountant: compute (ε, δ)-DP spent after `steps` applications of
the subsampled Gaussian mechanism with noise multiplier `σ` and sampling
rate `q`.

Uses the exact RDP bound for integer orders [Mironov et al. 2019]:

    ε_RDP(α) = (1/(α-1)) log( Σ_{k=0}^{α} C(α,k) (1-q)^{α-k} q^k
                                              exp(k(k-1)/(2σ²)) )

composed over `steps` via addition (RDP composition).  Converts to
(ε, δ)-DP via:
    ε = min_α { ε_RDP(α) + log(1/δ) / (α-1) }
"""
function _rdp_accountant(σ::Float64, q::Float64, steps::Int, delta::Float64)
    alphas = vcat(collect(2:10), collect(12:2:64), [128, 256])
    best_eps = Inf

    for α in alphas
        # Exact RDP for the subsampled Gaussian mechanism at integer α
        log_terms = Vector{Float64}(undef, α + 1)
        for k in 0:α
            log_binom = _logfactorial(α) - _logfactorial(k) - _logfactorial(α - k)
            log_coeff = log_binom +
                        k * log(max(q, 1e-300)) +
                        (α - k) * log(max(1 - q, 1e-300))
            log_moment = k * (k - 1) / (2.0 * σ^2)
            log_terms[k + 1] = log_coeff + log_moment
        end
        rdp = _logsumexp(log_terms) / (α - 1)
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
                       epochs, batch_size, privacy, rng, dev)
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

    # Move training data and params to device
    X_num_d     = _to_device(X_num, dev)
    X_cat_oh_d  = _to_device(X_cat_oh, dev)
    cat_indices_d = _to_device(cat_indices, dev)
    alphas_cumprod_d = _to_device(alphas_cumprod, dev)

    ps_all    = _to_device((; backbone = ps_bb, emb = ps_emb), dev)
    st_bb     = _to_device(st_bb, dev)
    st_emb    = _to_device(st_emb, dev)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps_all)

    t_start      = time()
    report_every = max(1, epochs ÷ 20)

    for epoch in 1:epochs
        epoch_loss = 0.0
        n_batches  = 0
        perm = Random.randperm(rng, nrows)
        for start in 1:batch_size:nrows
            stop = min(start + batch_size - 1, nrows)
            idx  = perm[start:stop]
            bs   = length(idx)

            t_batch = rand(rng, 1:T, bs)

            x_num_batch = d_num > 0 ? X_num_d[:, idx] : _to_device(zeros(Float32, 0, bs), dev)
            x_cat_batch = size(X_cat_oh_d, 1) > 0 ? X_cat_oh_d[:, idx] : _to_device(zeros(Float32, 0, bs), dev)

            if d_num > 0
                x_num_noised, ε = _gaussian_forward(x_num_batch, alphas_cumprod_d, t_batch, rng, dev)
            else
                x_num_noised = _to_device(zeros(Float32, 0, bs), dev)
                ε = _to_device(zeros(Float32, 0, bs), dev)
            end

            if size(x_cat_batch, 1) > 0
                x_cat_noised = _multinomial_forward(x_cat_batch, cat_dims, alphas_cumprod_d, t_batch, dev)
            else
                x_cat_noised = _to_device(zeros(Float32, 0, bs), dev)
            end

            # ── Per-sample gradient clipping ────────────────────────────
            gs_sum = _grad_zero(ps_all)
            batch_loss = 0.0

            for si in 1:bs
                xn_i  = d_num > 0 ? x_num_noised[:, si:si] : _to_device(zeros(Float32, 0, 1), dev)
                xc_i  = size(x_cat_noised, 1) > 0 ? x_cat_noised[:, si:si] : _to_device(zeros(Float32, 0, 1), dev)
                xc_orig_i = size(x_cat_batch, 1) > 0 ? x_cat_batch[:, si:si] : _to_device(zeros(Float32, 0, 1), dev)
                t_i_d = _to_device(Float32.([t_batch[si]]), dev)
                ε_i   = d_num > 0 ? ε[:, si:si] : _to_device(zeros(Float32, 0, 1), dev)

                l, _, g = _compute_grad(AD_BACKEND, ps_all) do p
                    _diffusion_loss(backbone, emb_layer,
                                    p.backbone, p.emb,
                                    st_bb, st_emb,
                                    xn_i, xc_i, xc_orig_i,
                                    t_i_d, ε_i, d_num, cat_dims)
                end
                batch_loss += l
                gnorm = sqrt(_grad_sqnorm(g))
                clip_factor = min(1.0, C / max(gnorm, 1e-12))
                g_clipped = _grad_scale(g, clip_factor)
                gs_sum = _grad_add(gs_sum, g_clipped)
            end

            epoch_loss += batch_loss / bs
            n_batches  += 1

            # Average and add noise
            gs_avg = _grad_scale(gs_sum, 1.0 / bs)
            noise_scale = σ_noise * C / bs
            gs_noisy = _grad_add_noise!(gs_avg, noise_scale, rng)

            opt_state, ps_all = Optimisers.update(opt_state, ps_all, gs_noisy)
        end

        # Progress report
        if epoch == 1 || epoch % report_every == 0 || epoch == epochs
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed  = time() - t_start
            eta      = elapsed / epoch * (epochs - epoch)
            @info "DP-SGD Epoch $(epoch)/$(epochs)  loss=$(round(avg_loss; digits=4))  elapsed=$(round(Int, elapsed))s  ETA=$(round(Int, eta))s"
        end
    end

    return ps_all.backbone, ps_all.emb, st_bb, st_emb
end

# ═══════════════════════════════════════════════════════════════════════════
# 11. Reverse Sampling (Denoising)
# ═══════════════════════════════════════════════════════════════════════════

"""
Sample `n` rows from a trained TabDDPM model via reverse denoising.

When `sampling_steps < T`, uses DDIM [Song et al. 2020] for the Gaussian
reverse step and the natural subsequence posterior for categoricals
[Hoogeboom et al. 2021].  `ddim_eta` controls stochasticity (0 =
deterministic DDIM, 1 ≈ DDPM variance).
"""
function _denoise_sample(backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
                         betas, alphas_cumprod, d_num, cat_dims, n, rng;
                         sampling_steps::Int = 0, ddim_eta::Float32 = 0f0)
    T = length(betas)
    d_cat_total = sum(cat_dims; init = 0)

    # Build timestep subsequence (1-indexed, descending)
    S = sampling_steps > 0 ? min(sampling_steps, T) : T
    if S == T
        # Full schedule: [T, T-1, …, 1]
        timesteps = collect(T:-1:1)
    else
        # Uniformly-spaced subsequence from the training schedule
        # E.g. S=50, T=1000 → [1000, 980, 960, …, 20]
        timesteps = reverse(round.(Int, range(1, T; length = S)))
        # Ensure no duplicates and the endpoints are included
        timesteps = unique(timesteps)
        S = length(timesteps)
    end

    # Detect device from params (they're already on the target device)
    gdev, cdev = _get_devices()

    # Initialize from pure noise on the compute device
    x_num = d_num > 0 ? _to_device(randn(rng, Float32, d_num, n), gdev) :
                        _to_device(zeros(Float32, 0, n), gdev)
    x_cat = d_cat_total > 0 ? begin
        # Uniform initialization for categoricals
        oh = zeros(Float32, d_cat_total, n)
        offset = 0
        for K in cat_dims
            oh[(offset + 1):(offset + K), :] .= 1f0 / K
            offset += K
        end
        _to_device(oh, gdev)
    end : _to_device(zeros(Float32, 0, n), gdev)

    for step_idx in 1:S
        t    = timesteps[step_idx]
        # ᾱ at the destination (previous) timestep
        ᾱ_prev = step_idx < S ? alphas_cumprod[timesteps[step_idx + 1]] : 1f0
        ᾱ_t    = alphas_cumprod[t]

        # Move timesteps to device as Float32 (match st_emb.freqs device)
        t_batch = _to_device(Float32.(fill(t, n)), gdev)

        # Timestep embedding (runs on device since ps_emb/st_emb are there)
        t_emb, _ = Lux.apply(emb_layer, t_batch, ps_emb, st_emb)

        # Feature input (timestep handled inside backbone via addition)
        features = if d_num > 0 && d_cat_total > 0
            vcat(x_num, x_cat)
        elseif d_num > 0
            x_num
        else
            x_cat
        end

        output, _ = Lux.apply(backbone, (features, t_emb), ps_bb, st_bb)

        # ── Gaussian reverse step (DDIM, Song et al. 2020) ─────────
        if d_num > 0
            ε_pred = output[1:d_num, :]

            # Predict x̂₀ from ε-prediction
            x0_pred = (x_num .- sqrt(1f0 - ᾱ_t + 1f-8) .* ε_pred) ./
                      sqrt(ᾱ_t + 1f-8)

            if step_idx < S   # not the final step
                # DDIM variance: σ² = η² · (1-ᾱ_prev)/(1-ᾱ_t) · (1-ᾱ_t/ᾱ_prev)
                σ² = ddim_eta^2 *
                     ((1f0 - ᾱ_prev) / (1f0 - ᾱ_t + 1f-8)) *
                     (1f0 - ᾱ_t / (ᾱ_prev + 1f-8))
                σ  = sqrt(max(σ², 0f0))

                # Direction pointing to x_t
                dir_coef = sqrt(max(1f0 - ᾱ_prev - σ², 0f0))

                x_num = sqrt(ᾱ_prev) .* x0_pred .+
                        dir_coef .* ε_pred

                if σ > 0f0
                    z = _to_device(randn(rng, Float32, d_num, n), gdev)
                    x_num = x_num .+ σ .* z
                end
            else
                # Final step: deterministic
                x_num = x0_pred
            end
        end

        # ── Multinomial reverse step (Hoogeboom et al. 2021) ────────
        # Posterior: q(x_prev|x_t, x̂₀) ∝ q(x_t|x_prev) · q(x_prev|x̂₀)
        # For skipped steps, α̃ = ᾱ_t/ᾱ_prev (effective retention).
        # Sampled via the Gumbel-max trick (GPU-native, no CPU transfer).
        if d_cat_total > 0
            # Effective single-step retention for the skip
            α_eff = ᾱ_t / (ᾱ_prev + 1f-20)

            offset_in  = d_num
            offset_out = 0
            for K in cat_dims
                logits = output[(offset_in + 1):(offset_in + K), :]

                # Predict x̂₀ probabilities from logits (softmax)
                m = maximum(logits; dims = 1)
                exp_logits = exp.(logits .- m)
                probs_x0 = exp_logits ./ sum(exp_logits; dims = 1)

                # Current one-hot state x_t for this variable
                x_t_block = x_cat[(offset_out + 1):(offset_out + K), :]

                # Likelihood: q(x_t | x_prev=k) = α̃·[x_t]_k + (1−α̃)/K
                log_likelihood = log.(α_eff .* x_t_block .+ (1f0 - α_eff) / K .+ 1f-20)
                # Prior:      q(x_prev=k | x̂₀)  = ᾱ_prev·[x̂₀]_k + (1−ᾱ_prev)/K
                log_prior = log.(ᾱ_prev .* probs_x0 .+ (1f0 - ᾱ_prev) / K .+ 1f-20)

                # Unnormalized log-posterior (normalization is unnecessary
                # for Gumbel-max since argmax is shift-invariant)
                log_unnorm = log_likelihood .+ log_prior

                # Gumbel-max trick: k = argmax(log p_k + Gumbel(0,1)) ~ Cat(p)
                u = _to_device(rand(rng, Float32, K, n), gdev)
                gumbel = -log.(-log.(u .+ 1f-10) .+ 1f-10)
                noisy = log_unnorm .+ gumbel

                # One-hot from column-wise argmax (fully on GPU)
                col_max = maximum(noisy; dims = 1)
                new_oh = Float32.(noisy .== col_max)

                # In-place update (not in AD-traced code, mutation is safe)
                x_cat[(offset_out + 1):(offset_out + K), :] .= new_oh

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
    embed_dim = gen.embed_dim
    hidden    = gen.hidden_dim > 0 ? gen.hidden_dim : min(256, max(64, 4 * (d_num + d_cat_total)))
    n_blocks  = gen.n_blocks
    d_in      = d_num + d_cat_total   # timestep handled via addition, not concat

    backbone, _ = _build_model(d_in, d_num, cat_dims_v;
                               hidden = hidden, n_blocks = n_blocks,
                               embed_dim = embed_dim, dropout = gen.dropout)
    emb_layer = SinusoidalEmbedding(embed_dim)

    lux_rng = Random.MersenneTwister(42)  # deterministic init
    ps_bb, st_bb   = Lux.setup(lux_rng, backbone)
    ps_emb, st_emb = Lux.setup(lux_rng, emb_layer)

    # ── Detect device (GPU if available) ─────────────────────────────
    gdev, cdev = _get_devices()
    @info "DiffusionGenerator: training on $(gdev)"

    # ── Train ──────────────────────────────────────────────────────────
    if gen.dp
        ps_bb, ps_emb, st_bb, st_emb = _train_dpsgd!(
            backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
            X_num, X_cat_oh, cat_indices,
            betas, alphas_cumprod, cat_dims_v, d_num,
            gen.epochs, gen.batch_size, privacy, rng, gdev)
    else
        ps_bb, ps_emb, st_bb, st_emb = _train_standard!(
            backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
            X_num, X_cat_oh, cat_indices,
            betas, alphas_cumprod, cat_dims_v, d_num,
            gen.epochs, gen.batch_size, rng, gdev)
    end

    # ── Move trained params back to CPU for storage / serialization ──
    ps_bb  = _to_device(ps_bb, cdev)
    ps_emb = _to_device(ps_emb, cdev)
    st_bb  = _to_device(st_bb, cdev)
    st_emb = _to_device(st_emb, cdev)

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

"""
    sample(model::FittedDiffusionModel, n; rng, sampling_steps, ddim_eta)

Generate `n` synthetic rows.  `sampling_steps` (default: full T=1000)
selects how many reverse-diffusion steps to run — fewer steps is faster
at some quality cost.  `ddim_eta` (default 0, deterministic DDIM)
controls stochasticity when step-skipping [Song et al. 2020].
"""
function sample(model::FittedDiffusionModel, n::Int;
                rng::AbstractRNG = model.rng,
                sampling_steps::Int = 0,
                ddim_eta::Float64 = 0.0)
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

    d_num     = length(model.num_columns)
    cat_dims  = model.cat_dims

    # ── Move model to best available device for sampling ───────────────
    gdev, cdev = _get_devices()
    ps_bb  = _to_device(full_ps.backbone, gdev)
    ps_emb = _to_device(full_ps.emb, gdev)
    st_bb  = _to_device(full_st.backbone, gdev)
    st_emb = _to_device(full_st.emb, gdev)

    # ── Reverse denoise ────────────────────────────────────────────────
    x_num, x_cat = _denoise_sample(
        backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
        model.betas, model.alphas_cumprod,
        d_num, cat_dims, n, rng;
        sampling_steps = sampling_steps,
        ddim_eta = Float32(ddim_eta))

    # Move results back to CPU for post-processing
    x_num = _to_device(x_num, cdev)
    x_cat = _to_device(x_cat, cdev)

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
