# ─── DataMimicLuxExt ────────────────────────────────────────────────────────
#
# Package extension: DiffusionGenerator engine (TabDDPM).
#
# Loaded when both Lux.jl and Zygote.jl are present.
# Implements REQ-DIF-001 through REQ-DIF-009.
#
# References:
#   [Kotelnikov et al. 2023]   — TabDDPM architecture (plain rtdl MLP with
#                                 additive timestep + label conditioning;
#                                 no normalization, no residual connections)
#   [Ho et al. 2020]           — DDPM (Gaussian diffusion)
#   [Song et al. 2020]         — DDIM (deterministic reverse sampling)
#   [Hoogeboom et al. 2021]    — Multinomial diffusion
#   [Abadi et al. 2016]        — DP-SGD
#   [Mironov et al. 2019]      — RDP of the Poisson-subsampled Gaussian

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
import SpecialFunctions

# Custom relu that bypasses cuDNN (avoids CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
# when cuDNN is missing or version-mismatched).  Pure CUDA element-wise kernel.
_fast_relu(x) = max(x, zero(x))

# SiLU / Swish activation — used in timestep embedding MLP and ResNet blocks
# [Kotelnikov et al. 2023, TabDDPM architecture].
_fast_silu(x) = x / (1f0 + exp(-x))

# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Preprocessing — Gaussian Quantile Transform
# ═══════════════════════════════════════════════════════════════════════════
#
# Following [Kotelnikov et al. 2023, TabDDPM] §4.1: continuous features are
# transformed via the Gaussian quantile method — each feature is mapped to
# approximate N(0,1) through its empirical CDF → Φ⁻¹.  This handles
# heavy-tailed distributions (e.g. capital-gain in Adult) far better than
# z-score normalization.

"""
    _quantile_forward(v, reference, n_ref) → Float32

Map a single value `v` to its Gaussian-quantile-transformed score using
sorted `reference` training values.  Returns Φ⁻¹(rank / (n+1)).
"""
function _quantile_forward(v::Float32, reference::Vector{Float32}, n_ref::Int)
    # Find position via binary search; average rank for ties
    lo = searchsortedfirst(reference, v)
    hi = searchsortedlast(reference, v)
    avg_rank = (lo + hi) / 2.0
    # Plotting position: (rank - 0.5) / n  ∈ (0, 1)
    q = clamp((avg_rank - 0.5) / n_ref, 1e-7, 1.0 - 1e-7)
    # Inverse normal CDF: Φ⁻¹(q) = √2 · erfinv(2q − 1)
    return Float32(sqrt(2.0) * SpecialFunctions.erfinv(2.0 * q - 1.0))
end

"""
    _quantile_inverse(z, reference) → Float32

Inverse Gaussian quantile transform: map a standard-normal value `z` back
to the original feature scale via Φ(z) → interpolation in sorted reference.
"""
function _quantile_inverse(z::Float32, reference::Vector{Float32})
    n = length(reference)
    # A diverged reverse process can hand back Inf or NaN.  Callers are
    # expected to discard those rows, but guard here so the failure surfaces
    # as a discardable extreme rather than as `InexactError` from
    # `floor(Int, NaN)` several frames down.
    isfinite(z) || return isnan(z) ? reference[(n + 1) ÷ 2] :
                          (z > 0 ? reference[end] : reference[1])
    # Normal CDF: Φ(z) = 0.5 · (1 + erf(z / √2))
    q = 0.5 * (1.0 + SpecialFunctions.erf(Float64(z) / sqrt(2.0)))
    q = clamp(q, 0.5 / n, 1.0 - 0.5 / n)
    # Map quantile to 1-indexed position in reference
    pos = q * n + 0.5
    lo = clamp(floor(Int, pos), 1, n)
    hi = clamp(lo + 1, 1, n)
    frac = Float32(pos - lo)
    return reference[lo] * (1f0 - frac) + reference[hi] * frac
end

"""
Pack table columns into Float32 arrays for training.

Returns `(X_num, X_cat_onehot, cat_indices, preprocess_info)`.
- `X_num`: `(d_num, N)` Float32 matrix, Gaussian-quantile normalized.
- `X_cat_onehot`: `(sum(cat_dims), N)` Float32 matrix.
- `cat_indices`: `(n_cat, N)` Int matrix — original category index per row.

When `target` names a column, it is held out of the categorical features and
returned separately as class labels for conditional generation.
"""
function _preprocess(cols, col_names, col_kinds, id_set, hints, nrows;
                     target::Union{Symbol, Nothing} = nothing)
    hint_dict = Dict(h.name => h for h in hints)

    num_cols       = Symbol[]
    cat_cols       = Symbol[]
    num_references = Vector{Float32}[]   # sorted training values per column
    num_round      = Bool[]              # round to integers on inverse transform
    cat_levels     = Dict{Symbol, Vector}()
    cat_dims       = Int[]

    # Class labels for conditional generation
    target_levels = Any[]
    y_indices     = Int[]

    for (i, name) in enumerate(col_names)
        kind = col_kinds[i]
        kind == :identifier && continue

        if name === target
            col = Tables.getcolumn(cols, name)
            nm  = DataMimic._nonmissing(collect(col))
            n_missing = nrows - length(nm)
            if n_missing > 0
                throw(ArgumentError(
                    "target column :$name has $n_missing missing value(s). " *
                    "Class-conditional generation needs a label for every row — " *
                    "drop those rows, impute them, or encode missing as an " *
                    "explicit level before fitting."))
            end
            target_levels = sort(unique(nm))
            lvl_map = Dict(v => k for (k, v) in enumerate(target_levels))
            y_indices = [lvl_map[col[r]] for r in 1:nrows]
            continue
        end

        if kind in (:continuous, :integer)
            push!(num_cols, name)
            nm = Float32.(DataMimic._nonmissing(collect(Tables.getcolumn(cols, name))))
            push!(num_references, sort(nm))
            # The reference implementation rounds a numeric column back to
            # integers when the training values are integral and few-valued.
            uniq = unique(nm)
            push!(num_round, length(uniq) <= 32 && all(v -> v == round(v), uniq))
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

    # ── Build numeric matrix (Gaussian quantile transform) ──────────────
    X_num = zeros(Float32, d_num, nrows)
    for (j, name) in enumerate(num_cols)
        col = Tables.getcolumn(cols, name)
        ref = num_references[j]
        n_ref = length(ref)
        for i in 1:nrows
            v = col[i]
            if ismissing(v)
                X_num[j, i] = 0f0   # missing → median of N(0,1)
            else
                X_num[j, i] = _quantile_forward(Float32(v), ref, n_ref)
            end
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

    # Empirical class distribution for conditional sampling
    class_dist = if isempty(target_levels)
        Float64[]
    else
        counts = zeros(Float64, length(target_levels))
        for k in y_indices
            counts[k] += 1
        end
        counts ./ sum(counts)
    end

    info = (; num_cols, cat_cols, num_references, num_round,
              cat_levels, cat_dims, d_num, d_cat_total,
              target_levels, y_indices, class_dist)
    return X_num, X_cat_oh, cat_indices, info
end

# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise Schedule
# ═══════════════════════════════════════════════════════════════════════════

"""
Cosine β schedule [Nichol & Dhariwal 2021], the TabDDPM default.

    ᾱ(t) = cos²((t + 0.008) / 1.008 · π/2),  β_i = min(1 - ᾱ(t₂)/ᾱ(t₁), 0.999)
"""
function _cosine_schedule(T::Int)
    ᾱ(t) = cos((t + 0.008) / 1.008 * π / 2)^2
    betas = Float32[min(1 - ᾱ((i + 1) / T) / ᾱ(i / T), 0.999) for i in 0:(T - 1)]
    alphas = 1f0 .- betas
    alphas_cumprod = cumprod(alphas)
    return betas, alphas_cumprod
end

# ── Log-domain helpers for multinomial diffusion ──────────────────────────

"""
Numerically stable `log(exp(a) + exp(b))`.

Guards the `a == b == -Inf` case, where `a - max(a, b)` would be `NaN`.
"""
function _log_add_exp(a, b)
    m = max(a, b)
    isfinite(m) || return m
    return m + log(exp(a - m) + exp(b - m))
end

"""
`log(1 - exp(a))` for `a ≤ 0`, evaluated without catastrophic cancellation.

The schedule's cumulative α values sit very close to 1 at small `t`, so `a` is
very close to 0 and the naive `log(1 - exp(a))` loses most of its significant
digits.  Two branches keep the relative error small across the range
[Mächler 2012, "Accurately Computing log(1 - exp(-|a|))"].
"""
function _log_1_min_a(a)
    a >= 0 && return oftype(float(a), -Inf)
    return a > -log(oftype(float(a), 2)) ? log(-expm1(a)) : log1p(-exp(a))
end

"""
Derive every schedule constant the Gaussian and multinomial processes need.

Returns a NamedTuple with the Gaussian posterior coefficients and the
log-domain α terms used by multinomial diffusion [Hoogeboom et al. 2021].
"""
function _schedule_constants(betas_f32::Vector{Float32})
    # Derived in Float64 and narrowed at the end, as the reference does.  The
    # cumulative products run very close to 1 at small t and to 0 at large t,
    # so accumulating them in Float32 would lose most of the precision.
    T = length(betas_f32)
    betas  = Float64.(betas_f32)
    alphas = 1.0 .- betas
    alphas_cumprod = cumprod(alphas)
    alphas_cumprod_prev = vcat(1.0, alphas_cumprod[1:(end - 1)])

    # Gaussian posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas .* (1.0 .- alphas_cumprod_prev) ./ (1.0 .- alphas_cumprod)
    posterior_log_variance_clipped = log.(vcat(posterior_variance[2], posterior_variance[2:end]))
    posterior_mean_coef1 = betas .* sqrt.(alphas_cumprod_prev) ./ (1.0 .- alphas_cumprod)
    posterior_mean_coef2 = (1.0 .- alphas_cumprod_prev) .* sqrt.(alphas) ./ (1.0 .- alphas_cumprod)

    # Multinomial (log domain)
    log_alpha = log.(alphas)
    log_cumprod_alpha = cumsum(log_alpha)
    log_1_min_alpha = _log_1_min_a.(log_alpha)
    log_1_min_cumprod_alpha = _log_1_min_a.(log_cumprod_alpha)

    f32(x) = Float32.(x)
    return (; betas = betas_f32,
              alphas = f32(alphas),
              alphas_cumprod = f32(alphas_cumprod),
              alphas_cumprod_prev = f32(alphas_cumprod_prev),
              posterior_variance = f32(posterior_variance),
              posterior_log_variance_clipped = f32(posterior_log_variance_clipped),
              posterior_mean_coef1 = f32(posterior_mean_coef1),
              posterior_mean_coef2 = f32(posterior_mean_coef2),
              log_alpha = f32(log_alpha),
              log_cumprod_alpha = f32(log_cumprod_alpha),
              log_1_min_alpha = f32(log_1_min_alpha),
              log_1_min_cumprod_alpha = f32(log_1_min_cumprod_alpha), T)
end

"""
Row offsets delimiting each categorical block in a stacked one-hot matrix.
`cat_dims = [2, 3]` → `[0, 2, 5]`.
"""
_cat_offsets(cat_dims::Vector{Int}) = cumsum(vcat(0, cat_dims))

# ── Ragged categorical blocks as one padded tensor ────────────────────────
#
# Categorical features have different cardinalities, so the natural way to
# normalize them is a loop over blocks.  That is fatal here: the block ops sit
# inside the differentiated loss, so Zygote generates a separate pullback per
# block, three times over (`_predict_start`, and `_q_posterior` both directly
# and via `_p_pred`).  On a table with nine categorical columns that is ~27
# block-pullbacks inlined into one function, and compilation blows up
# superlinearly — minutes to compile a single gradient.
#
# Padding the blocks into a rectangular (K_max, n_cat, batch) tensor turns
# every per-block reduction into one reduction along dimension 1.  The op count
# is then constant in the number of categorical columns.  Padding uses a large
# negative sentinel rather than -Inf so that `exp` underflows cleanly to zero
# without risking NaN in the pullback.

const _PAD_SENTINEL = -1f30

"""
Precompute the block-structure operators for a set of categorical dimensions.

- `B` / `Bt`: `(n_cat, d_cat)` membership matrix and its transpose.  Summing
  within each block is linear, so `Bt * (B * e)` computes every block sum and
  broadcasts it back over the block's rows in two cuBLAS matmuls — far faster
  than index-based gathers, and its adjoint is another matmul.
- `P` / `U`: pad/unpad operators to the `(K_max, n_cat, batch)` layout, needed
  only for the per-block argmax in sampling (max is not linear, so it cannot
  be expressed as a matmul).
"""
function _block_plan(cat_dims::Vector{Int}, dev)
    n_cat = length(cat_dims)
    n_cat == 0 && return nothing
    K_max = maximum(cat_dims)
    d_cat = sum(cat_dims)
    offs  = _cat_offsets(cat_dims)
    pad_row = d_cat + 1                      # sentinel row appended to x

    B = zeros(Float32, n_cat, d_cat)
    for c in 1:n_cat, k in 1:cat_dims[c]
        B[c, offs[c] + k] = 1f0
    end

    P = zeros(Float32, K_max * n_cat, d_cat + 1)
    for c in 1:n_cat, k in 1:K_max
        P[(c - 1) * K_max + k, k <= cat_dims[c] ? offs[c] + k : pad_row] = 1f0
    end
    U = zeros(Float32, d_cat, K_max * n_cat)
    for c in 1:n_cat, k in 1:cat_dims[c]
        U[offs[c] + k, (c - 1) * K_max + k] = 1f0
    end

    return (; n_cat, K_max, d_cat,
              B  = _to_device(B, dev),
              Bt = _to_device(collect(B'), dev),
              P  = _to_device(P, dev),
              U  = _to_device(U, dev))
end

"""Pad the packed categorical rows into the `(K_max, n_cat, batch)` layout."""
function _to_padded(x, plan)
    pad = @view(x[1:1, :]) .* 0f0 .+ _PAD_SENTINEL
    return reshape(plan.P * vcat(x, pad), plan.K_max, plan.n_cat, :)
end

"""Unpad back to the packed `(Σ K, batch)` layout."""
_from_padded(padded, plan) =
    plan.U * reshape(padded, plan.K_max * plan.n_cat, :)

"""
Normalize each categorical block to log-probabilities (per-block log-softmax).

Equivalent to `x - sliced_logsumexp(x)` in the reference implementation.

The per-block sum is computed as two matmuls against the block-membership
matrix, which keeps this to a handful of cuBLAS/elementwise kernels.  A single
global maximum is enough to stabilize `exp`: these are log-probabilities, so
the shifted values stay well inside Float32 range.
"""
function _block_log_normalize(x, plan)
    z = x .- maximum(x)
    s = plan.Bt * (plan.B * exp.(z))     # block sums, broadcast back over rows
    return z .- log.(s)
end

"""KL divergence between two per-feature categorical log-probability matrices."""
_multinomial_kl(log_p, log_q) = sum(exp.(log_p) .* (log_p .- log_q); dims = 1)

"""log q(x_t | x_0) — the `q_pred` of the reference implementation."""
function _q_pred(log_x_start, log_cumprod_alpha_t, log_1_min_cumprod_alpha_t, log_K)
    return _log_add_exp.(log_x_start .+ log_cumprod_alpha_t,
                         log_1_min_cumprod_alpha_t .- log_K)
end

"""log q(x_t | x_{t-1}) — one-step forward, `q_pred_one_timestep`."""
function _q_pred_one_step(log_x_t, log_alpha_t, log_1_min_alpha_t, log_K)
    return _log_add_exp.(log_x_t .+ log_alpha_t,
                         log_1_min_alpha_t .- log_K)
end

# ── Per-batch schedule coefficients ───────────────────────────────────────
#
# The multinomial process needs four schedule values per row, plus a mask for
# the `t == 1` boundary.  Indexing the schedule on the host and copying the
# slices across for every loss call costs one synchronization per transfer and
# dominates the step time, so the schedule is kept device-resident and gathered
# with a device-side index vector.  Sampling uses a single scalar timestep, so
# its coefficients are plain scalars that broadcast for free.

"""Move the schedule arrays the multinomial branch indexes onto `dev`."""
function _device_schedule(sched, dev)
    return (; log_alpha               = _to_device(sched.log_alpha, dev),
              log_1_min_alpha         = _to_device(sched.log_1_min_alpha, dev),
              log_cumprod_alpha       = _to_device(sched.log_cumprod_alpha, dev),
              log_1_min_cumprod_alpha = _to_device(sched.log_1_min_cumprod_alpha, dev))
end

"""Gather the per-row schedule coefficients for a training batch (on device)."""
function _batch_coefs(sched_d, t_d)
    t_prev = max.(t_d .- 1, 1)
    return (; lca_prev = reshape(sched_d.log_cumprod_alpha[t_prev], 1, :),
              l1m_prev = reshape(sched_d.log_1_min_cumprod_alpha[t_prev], 1, :),
              lca      = reshape(sched_d.log_cumprod_alpha[t_d], 1, :),
              l1m      = reshape(sched_d.log_1_min_cumprod_alpha[t_d], 1, :),
              la       = reshape(sched_d.log_alpha[t_d], 1, :),
              l1a      = reshape(sched_d.log_1_min_alpha[t_d], 1, :),
              is_first = reshape(Float32.(t_d .== 1), 1, :))
end

"""Schedule coefficients for a single scalar timestep (sampling)."""
function _scalar_coefs(sched, t::Int)
    tp = max(t - 1, 1)
    return (; lca_prev = sched.log_cumprod_alpha[tp],
              l1m_prev = sched.log_1_min_cumprod_alpha[tp],
              lca      = sched.log_cumprod_alpha[t],
              l1m      = sched.log_1_min_cumprod_alpha[t],
              la       = sched.log_alpha[t],
              l1a      = sched.log_1_min_alpha[t],
              is_first = t == 1 ? 1f0 : 0f0)
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
    # cos before sin, matching `timestep_embedding` in the reference code
    emb    = vcat(cos.(args), sin.(args))  # (dim, batch)
    return emb, st
end

Lux.statelength(l::SinusoidalEmbedding) = l.dim ÷ 2

# ═══════════════════════════════════════════════════════════════════════════
# 4. TabDDPM Backbone  [Kotelnikov et al. 2023]
# ═══════════════════════════════════════════════════════════════════════════
#
# Mirrors `MLPDiffusion` in the reference implementation:
#   emb = time_embed(timestep_embedding(t, dim_t))
#   emb += silu(label_emb(y))            # only when class-conditional
#   x   = proj(x) + emb
#   out = mlp(x)
#
# `mlp` is the rtdl baseline MLP [Gorishniy et al. 2021]: a stack of
# `Dense → ReLU → Dropout` blocks followed by a linear head.  There is no
# normalization and there are no residual connections.

"""
Lux container layer for the TabDDPM denoising network [Kotelnikov et al. 2023].

Subcomponents:
- `proj`:       Dense(d_in → dim_t) — input feature projection
- `time_embed`: Chain(Dense → SiLU → Dense) — timestep embedding projection
- `label_emb`:  Embedding(n_classes → dim_t), or `NoOpLayer` when unconditional
- `mlp`:        Chain of Dense(ReLU) + Dropout blocks, then a linear head

Call signature: `(model)((features, t_emb, y), ps, st)` where
- `features`: (d_in, batch) — numeric ⧺ log-one-hot categorical features
- `t_emb`:    (dim_t, batch) — raw sinusoidal timestep embedding
- `y`:        (batch,) integer class indices, or `nothing` when unconditional
"""
struct TabDDPMBackbone <: Lux.AbstractLuxContainerLayer{(:proj, :time_embed, :label_emb, :mlp)}
    proj
    time_embed
    label_emb
    mlp
end

function (m::TabDDPMBackbone)((features, t_emb, y), ps, st)
    # Timestep conditioning: (dim_t, B) → (dim_t, B)
    emb, st_time = Lux.apply(m.time_embed, t_emb, ps.time_embed, st.time_embed)

    # Class conditioning: emb += silu(label_emb(y))
    st_label = st.label_emb
    if y !== nothing && !(m.label_emb isa Lux.NoOpLayer)
        lab, st_label = Lux.apply(m.label_emb, y, ps.label_emb, st.label_emb)
        emb = emb .+ _fast_silu.(lab)
    end

    # Project features and add the conditioning vector
    h, st_proj = Lux.apply(m.proj, features, ps.proj, st.proj)
    h = h .+ emb

    output, st_mlp = Lux.apply(m.mlp, h, ps.mlp, st.mlp)

    st_new = (; proj = st_proj, time_embed = st_time,
                label_emb = st_label, mlp = st_mlp)
    return output, st_new
end

"""
Build the TabDDPM denoising network [Kotelnikov et al. 2023].

    proj(features) + time_embed(t_emb) [+ silu(label_emb(y))] → MLP → output

- `d_in`:      feature dimension (d_num + Σ cat_dims)
- `d_num`:     number of numeric output channels (predicts noise ε)
- `cat_dims`:  per-categorical output logits
- `d_layers`:  per-layer widths of the MLP (the paper's `rtdl_params.d_layers`)
- `embed_dim`: timestep embedding width (`dim_t`)
- `n_classes`: number of label classes; 0 = unconditional
"""
function _build_model(d_in::Int, d_num::Int, cat_dims::Vector{Int};
                      d_layers::Vector{Int} = [256, 256],
                      embed_dim::Int = 128, dropout::Float64 = 0.0,
                      n_classes::Int = 0)
    d_out = d_num + sum(cat_dims; init = 0)

    proj = Dense(d_in => embed_dim)

    # time_embed: Linear(dim_t, dim_t) → SiLU → Linear(dim_t, dim_t)
    time_embed = Chain(
        Dense(embed_dim => embed_dim, _fast_silu),
        Dense(embed_dim => embed_dim))

    label_emb = n_classes > 0 ? Lux.Embedding(n_classes => embed_dim) : Lux.NoOpLayer()

    # rtdl baseline MLP: [Dense(ReLU) → Dropout] per layer, then a linear head.
    #
    # Zygote's compile time is superlinear in `Chain` length (measured: a
    # 2-layer stack compiles in 0.8s, a 4-layer stack in 17.6s), so the
    # zero-probability Dropout layers are omitted rather than included as
    # no-ops — they would double the chain length for no numerical effect.
    p = Float32(max(dropout, 0.0))
    layers = Any[]
    d_prev = embed_dim
    for d in d_layers
        push!(layers, Dense(d_prev => d, _fast_relu))
        p > 0 && push!(layers, Lux.Dropout(p))
        d_prev = d
    end
    push!(layers, Dense(d_prev => d_out))
    mlp = Chain(layers...)

    model = TabDDPMBackbone(proj, time_embed, label_emb, mlp)
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

# ── Multinomial diffusion in log space [Hoogeboom et al. 2021] ────────────
#
# The reference implementation carries categorical state as *log* one-hot
# vectors throughout: the network input, the forward process, and the reverse
# posterior all operate on log-probabilities.

"""Row vector of `log K` repeated across each categorical block."""
function _log_K_vector(cat_dims::Vector{Int})
    return vcat([fill(Float32(log(K)), K) for K in cat_dims]...)
end

"""Convert a stacked one-hot matrix to log space (`index_to_log_onehot`)."""
_to_log_onehot(x_oh) = log.(max.(x_oh, 1f-30))

"""
Draw categories via the Gumbel-max trick, returning a log-one-hot matrix.

Mirrors `log_sample_categorical`: independent Gumbel noise per category,
argmax within each categorical block.
"""
function _log_sample_categorical(logits, plan, rng, dev)
    u = _to_device(rand(rng, Float32, size(logits)...), dev)
    gumbel = -log.(-log.(u .+ 1f-30) .+ 1f-30)
    noisy  = gumbel .+ logits

    # Per-block argmax as a one-hot mask, computed on-device in one reduction
    # (ties have probability zero under continuous Gumbel noise, and the
    # padding sentinel can never be the block maximum).
    padded = _to_padded(noisy, plan)
    mask = Float32.(padded .== maximum(padded; dims = 1))
    return _to_log_onehot(_from_padded(mask, plan))
end

"""
Multinomial forward diffusion: `q(x_t | x_0)` followed by a categorical draw.

Returns the log-one-hot state at timestep `t`.
"""
function _multinomial_q_sample(log_x_start, plan, coef, log_K, rng, dev)
    log_probs = _q_pred(log_x_start, coef.lca, coef.l1m, log_K)
    return _log_sample_categorical(log_probs, plan, rng, dev)
end

"""
Per-categorical-block log-softmax of the network output (`predict_start`).

The multinomial branch uses the `x0` parametrization: the network predicts
log-probabilities of the *clean* categories.
"""
_predict_start(model_out_cat, plan) = _block_log_normalize(model_out_cat, plan)

"""
Log of the true reverse posterior `q(x_{t-1} | x_t, x_0)` (`q_posterior`).

`t_idx` is the 1-based timestep vector; `t_prev` is `max(t-1, 1)` with the
`t == 1` entries replaced by `log_x_start`, matching the reference guard.
"""
function _q_posterior(log_x_start, log_x_t, plan, coef, log_K)
    log_EV = _q_pred(log_x_start, coef.lca_prev, coef.l1m_prev, log_K)

    # At t == 1 the posterior is exactly x_0
    log_EV = coef.is_first .* log_x_start .+ (1f0 .- coef.is_first) .* log_EV

    unnormed = log_EV .+ _q_pred_one_step(log_x_t, coef.la, coef.l1a, log_K)

    return _block_log_normalize(unnormed, plan)
end

"""Reverse model distribution `p(x_{t-1} | x_t)` (`p_pred`, x0 parametrization)."""
function _p_pred(model_out_cat, log_x_t, plan, coef, log_K)
    log_x_recon = _predict_start(model_out_cat, plan)
    return _q_posterior(log_x_recon, log_x_t, plan, coef, log_K)
end

# ═══════════════════════════════════════════════════════════════════════════
# 6. Loss Function
# ═══════════════════════════════════════════════════════════════════════════

"""
Prior KL term of the multinomial variational bound (`kl_prior`).

Compares `q(x_T | x_0)` against the uniform categorical prior.  Independent of
the network output, but part of the reported loss in the reference code.
"""
function _kl_prior(log_x_start, log_K, lca_T::Float32, l1m_T::Float32)
    log_qxT = _q_pred(log_x_start, lca_T, l1m_T, log_K)
    return _multinomial_kl(log_qxT, -log_K)
end

"""
TabDDPM mixed loss [Kotelnikov et al. 2023]: `loss_multi + loss_gauss`.

- Gaussian branch: MSE between true and predicted noise (`mean_flat`).
- Multinomial branch: the stochastic variational bound `L_t / p_t + KL_prior`,
  averaged over the batch and divided by the number of categorical features.

`log_x_cat_start` / `log_x_cat_t` are log-one-hot matrices; `t_idx` is the
1-based CPU timestep vector used to index the schedule.
"""
function _diffusion_loss(backbone, emb_layer, ps_backbone, ps_emb,
                         st_backbone, st_emb,
                         X_num_noised, log_x_cat_t, log_x_cat_start,
                         t_batch, coef, ε_true, d_num, plan,
                         n_timesteps, log_K, lca_T, l1m_T, y_batch)
    # Timestep embedding (sinusoidal)
    t_emb, st_emb_new = Lux.apply(emb_layer, t_batch, ps_emb, st_emb)

    features = if d_num > 0 && plan !== nothing
        vcat(X_num_noised, log_x_cat_t)
    elseif d_num > 0
        X_num_noised
    else
        log_x_cat_t
    end

    output, st_bb_new = Lux.apply(backbone, (features, t_emb, y_batch),
                                  ps_backbone, st_backbone)

    loss = 0f0

    # ── Gaussian branch: mean_flat((ε - ε̂)²) ───────────────────────────
    if d_num > 0
        ε_pred = output[1:d_num, :]
        loss += sum(abs2, ε_pred .- ε_true) / (d_num * size(output, 2))
    end

    # ── Multinomial branch: variational bound ──────────────────────────
    if plan !== nothing
        n_cat = plan.n_cat
        model_out_cat = output[(d_num + 1):end, :]

        log_true_prob  = _q_posterior(log_x_cat_start, log_x_cat_t,
                                      plan, coef, log_K)
        log_model_prob = _p_pred(model_out_cat, log_x_cat_t,
                                 plan, coef, log_K)

        kl = _multinomial_kl(log_true_prob, log_model_prob)               # (1, B)
        decoder_nll = -sum(exp.(log_x_cat_start) .* log_model_prob; dims = 1)

        # At t == 1 the bound uses the decoder likelihood instead of the KL
        Lt = coef.is_first .* decoder_nll .+ (1f0 .- coef.is_first) .* kl

        # Uniform timestep sampling ⇒ p_t = 1/T, so L_t / p_t = T · L_t
        vb = Lt .* Float32(n_timesteps) .+ _kl_prior(log_x_cat_start, log_K, lca_T, l1m_T)
        loss += sum(vb) / (n_cat * size(output, 2))
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
# 9. Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════════

"""
Linear learning-rate annealing, matching `_anneal_lr` in the reference code:

    lr = lr_max · (1 - step / total_steps)

`warmup` epochs, when requested, linearly ramp from `lr_max/10` to `lr_max`
before the anneal begins; with `warmup == 0` this is exactly the paper's
schedule.
"""
function _anneal_lr(epoch::Int, total_epochs::Int, lr_max::Float64, warmup::Int)
    if warmup > 0 && epoch <= warmup
        return lr_max * (0.1 + 0.9 * (epoch - 1) / max(warmup - 1, 1))
    end
    done = (epoch - warmup) / max(total_epochs - warmup, 1)
    return lr_max * max(1.0 - done, 0.0)
end

# ── Exponential moving average of the denoiser weights ────────────────────
#
# The reference implementation keeps an EMA copy of the denoiser and saves it
# alongside the raw weights; sampling uses the EMA copy.

"""
In-place EMA update `target ← rate · target + (1 - rate) · source`.

Mutates `target`, which is a private copy owned by the training loop and never
part of an AD trace — matching the reference implementation's `mul_`/`add_`
and avoiding a full parameter-tree allocation on every step.
"""
function _ema_update!(target, source, rate::Float32)
    Lux.Functors.fmap(target, source) do t, s
        t isa AbstractArray && (@. t = rate * t + (1f0 - rate) * s)
        t
    end
    return target
end

"""
Abort training when the loss stops being finite.

A diverged run reports `loss=NaN` and otherwise proceeds normally: gradients
are NaN, every weight becomes NaN, and the remaining epochs are wasted on a
model that is already dead.  Nothing surfaces until sampling fails much later,
by which point the cause is far away from the symptom.  Failing here instead
names the epoch it happened and what usually causes it.
"""
function _check_finite_loss(loss, epoch::Int, epochs::Int, lr)
    isfinite(loss) && return nothing
    error("DiffusionGenerator training diverged: loss became " *
          "$(isnan(loss) ? "NaN" : "Inf") at epoch $epoch of $epochs " *
          "(learning rate $(round(lr; sigdigits = 3))). Every weight is now " *
          "non-finite, so the remaining epochs cannot recover. Try a lower " *
          "`lr`, a smaller `batch_size`, fewer `num_timesteps`, or a narrower " *
          "`d_layers`.")
end

# ═══════════════════════════════════════════════════════════════════════════
# 10. Standard Training Loop
# ═══════════════════════════════════════════════════════════════════════════

function _train_standard!(backbone, emb_layer, ps_bb, ps_emb,
                          st_bb, st_emb,
                          X_num, X_cat_oh, y_indices,
                          sched, cat_dims, d_num,
                          epochs, batch_size, lr, lr_warmup,
                          weight_decay, ema_decay, rng, dev)
    T     = sched.T
    nrows = size(X_num, 2) > 0 ? size(X_num, 2) : size(X_cat_oh, 2)
    log_K = _to_device(_log_K_vector(cat_dims), dev)
    plan  = _block_plan(cat_dims, dev)

    # Move training data and params to device (GPU if available)
    X_num_d    = _to_device(X_num, dev)
    log_x_cat_all = size(X_cat_oh, 1) > 0 ?
        _to_device(_to_log_onehot(X_cat_oh), dev) : _to_device(X_cat_oh, dev)
    alphas_cumprod_d = _to_device(sched.alphas_cumprod, dev)
    sched_d = _device_schedule(sched, dev)
    lca_T = Float32(sched.log_cumprod_alpha[T])
    l1m_T = Float32(sched.log_1_min_cumprod_alpha[T])
    conditional = !isempty(y_indices)

    # Merge params for optimizer
    ps_all    = _to_device((; backbone = ps_bb, emb = ps_emb), dev)
    st_bb     = _to_device(st_bb, dev)
    st_emb    = _to_device(st_emb, dev)
    # AdamW, matching the reference implementation
    opt_state = Optimisers.setup(
        Optimisers.AdamW(Float32(lr), (0.9f0, 0.999f0), Float32(weight_decay)), ps_all)

    # EMA copy of the denoiser weights, used for sampling
    use_ema  = ema_decay > 0
    ema_rate = Float32(ema_decay)
    ps_ema   = use_ema ? Lux.Functors.fmap(x -> x isa AbstractArray ? copy(x) : x, ps_all) : ps_all

    t_start    = time()
    epoch_loss = 0.0
    n_batches  = 0

    # Progress reporting interval: ~20 updates over the full run
    report_every = max(1, epochs ÷ 20)

    for epoch in 1:epochs
        cur_lr = Float32(_anneal_lr(epoch, epochs, lr, lr_warmup))
        Optimisers.adjust!(opt_state; eta = cur_lr)

        epoch_loss = 0.0
        n_batches  = 0
        perm = Random.randperm(rng, nrows)
        for start in 1:batch_size:nrows
            stop = min(start + batch_size - 1, nrows)
            idx  = perm[start:stop]
            bs   = length(idx)

            t_batch = rand(rng, 1:T, bs)
            # One host→device copy of the timesteps; every schedule coefficient
            # is then gathered on-device from `sched_d`.
            t_d = _to_device(t_batch, dev)
            # Timestep embedding uses 0-based t, as in the reference code
            t_batch_d = Float32.(t_d .- 1)
            coef = _batch_coefs(sched_d, t_d)
            y_batch = conditional ? _to_device(y_indices[idx], dev) : nothing

            # Slice with a device-resident index: indexing a device array with a
            # host vector forces a slow host-driven gather (~20 ms per slice).
            idx_d = _to_device(idx, dev)

            # Forward diffusion (slice on device)
            x_num_batch = d_num > 0 ? X_num_d[:, idx_d] : _to_device(zeros(Float32, 0, bs), dev)

            if d_num > 0
                x_num_noised, ε = _gaussian_forward(x_num_batch, alphas_cumprod_d, t_batch, rng, dev)
            else
                x_num_noised = _to_device(zeros(Float32, 0, bs), dev)
                ε = _to_device(zeros(Float32, 0, bs), dev)
            end

            if plan !== nothing
                log_x_cat_start = log_x_cat_all[:, idx_d]
                log_x_cat_t = _multinomial_q_sample(log_x_cat_start, plan, coef,
                                                    log_K, rng, dev)
            else
                log_x_cat_start = _to_device(zeros(Float32, 0, bs), dev)
                log_x_cat_t     = log_x_cat_start
            end

            # Gradient (dispatched through AD backend — REQ-DIF-009)
            loss, states_new, g = _compute_grad(AD_BACKEND, ps_all) do p
                _diffusion_loss(backbone, emb_layer,
                                p.backbone, p.emb,
                                st_bb, st_emb,
                                x_num_noised, log_x_cat_t, log_x_cat_start,
                                t_batch_d, coef, ε, d_num, plan,
                                T, log_K, lca_T, l1m_T, y_batch)
            end
            st_bb, st_emb = states_new
            epoch_loss += loss
            n_batches  += 1

            opt_state, ps_all = Optimisers.update(opt_state, ps_all, g)
            use_ema && _ema_update!(ps_ema, ps_all, ema_rate)
        end

        avg_loss = epoch_loss / max(n_batches, 1)
        _check_finite_loss(avg_loss, epoch, epochs, cur_lr)

        # Progress report
        if epoch == 1 || epoch % report_every == 0 || epoch == epochs
            elapsed  = time() - t_start
            eta      = elapsed / epoch * (epochs - epoch)
            @info "Epoch $(epoch)/$(epochs)  loss=$(round(avg_loss; digits=4))  lr=$(round(cur_lr; sigdigits=3))  elapsed=$(round(Int, elapsed))s  ETA=$(round(Int, eta))s"
        end
    end

    ps_out = use_ema ? ps_ema : ps_all
    return ps_out.backbone, ps_out.emb, st_bb, st_emb
end

# ═══════════════════════════════════════════════════════════════════════════
# 11. DP-SGD Training Loop
# ═══════════════════════════════════════════════════════════════════════════

"""Log-sum-exp with numerical stability."""
function _logsumexp(xs)
    m = maximum(xs)
    isinf(m) && return m
    return m + log(sum(exp.(xs .- m)))
end

"""Log of n!, summed directly — the arguments here are small integer orders."""
function _logfactorial(n::Int)
    n <= 1 && return 0.0
    return sum(log(Float64(i)) for i in 2:n)
end

"""
Rényi DP accountant: bound the (ε, δ)-DP spend after `steps` applications of
the *Poisson-subsampled* Gaussian mechanism with noise multiplier `σ` and
sampling rate `q` (each record included independently with probability `q`).

For integer orders α the Rényi divergence of the sampled Gaussian mechanism
has the closed form [Mironov et al. 2019, Sec. 3.3]:

    ε_RDP(α) = (1/(α-1)) log( Σ_{k=0}^{α} C(α,k) (1-q)^{α-k} q^k
                                              exp(k(k-1)/(2σ²)) )

which is exact for that order (it is the dominating direction of the two
Rényi divergences defining the mechanism's RDP).  The reported ε is
nonetheless an *upper bound* rather than the tightest achievable value, for
two reasons:

  1. the order is minimized over a finite grid of integer α only — real
     orders in between, which can be slightly better, are not searched; and
  2. the RDP → (ε, δ) conversion below is the standard bound of
     [Mironov 2017, Prop. 3]; tighter conversions exist
     [Balle et al. 2020, Canonne et al. 2020].

Both approximations err on the conservative side, so the returned ε is a
valid (if not minimal) guarantee.  Composition over `steps` is by addition
of RDP at a common order, and the conversion is

    ε = min_α { steps · ε_RDP(α) + log(1/δ) / (α-1) }

This bound is only valid for **Poisson** subsampling — the caller must
sample minibatches accordingly (see `_train_dpsgd!`).
"""
function _rdp_accountant(σ::Float64, q::Float64, steps::Int, delta::Float64)
    alphas = vcat(collect(2:10), collect(12:2:64), [128, 256])
    best_eps = Inf

    for α in alphas
        # Closed-form RDP of the Poisson-subsampled Gaussian at integer α
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
                       X_num, X_cat_oh, y_indices,
                       sched, cat_dims, d_num,
                       epochs, batch_size, lr, lr_warmup,
                       weight_decay, privacy, rng, dev)
    T     = sched.T
    nrows = size(X_num, 2) > 0 ? size(X_num, 2) : size(X_cat_oh, 2)
    log_K = _to_device(_log_K_vector(cat_dims), dev)
    plan  = _block_plan(cat_dims, dev)
    sched_d = _device_schedule(sched, dev)
    lca_T = Float32(sched.log_cumprod_alpha[T])
    l1m_T = Float32(sched.log_1_min_cumprod_alpha[T])
    conditional = !isempty(y_indices)

    # DP-SGD parameters (REQ-DIF-005)
    C = 1.0                                     # gradient clip norm
    q = min(batch_size / nrows, 1.0)            # Poisson sampling rate
    # Expected lot size.  Both the gradient average and the noise scale are
    # normalized by this *constant*, never by the realized batch size — the
    # latter is data-dependent and would leak.
    expected_bs = q * nrows
    steps_per_epoch = ceil(Int, nrows / batch_size)
    total_steps = epochs * steps_per_epoch

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
    X_num_d    = _to_device(X_num, dev)
    log_x_cat_all = size(X_cat_oh, 1) > 0 ?
        _to_device(_to_log_onehot(X_cat_oh), dev) : _to_device(X_cat_oh, dev)
    alphas_cumprod_d = _to_device(sched.alphas_cumprod, dev)

    ps_all    = _to_device((; backbone = ps_bb, emb = ps_emb), dev)
    st_bb     = _to_device(st_bb, dev)
    st_emb    = _to_device(st_emb, dev)
    opt_state = Optimisers.setup(
        Optimisers.AdamW(Float32(lr), (0.9f0, 0.999f0), Float32(weight_decay)), ps_all)

    t_start      = time()
    report_every = max(1, epochs ÷ 20)

    for epoch in 1:epochs
        cur_lr = Float32(_anneal_lr(epoch, epochs, lr, lr_warmup))
        Optimisers.adjust!(opt_state; eta = cur_lr)

        epoch_loss = 0.0
        n_batches  = 0
        # Poisson subsampling (REQ-DIF-005): each record is included in the
        # lot independently with probability q.  This is exactly the
        # mechanism `_rdp_accountant` models; shuffle-and-partition over a
        # random permutation is a *different* mechanism with different
        # amplification, so it must not be used here.
        for _step in 1:steps_per_epoch
            idx = findall(rand(rng, nrows) .< q)
            bs  = length(idx)

            # An empty lot is a legitimate outcome of Poisson sampling.  It
            # still consumes a step of budget, so the (all-zero) gradient is
            # still noised and applied.
            if bs == 0
                gs_noisy = _grad_add_noise!(_grad_zero(ps_all),
                                            σ_noise * C / expected_bs, rng)
                opt_state, ps_all = Optimisers.update(opt_state, ps_all, gs_noisy)
                continue
            end

            t_batch = rand(rng, 1:T, bs)

            x_num_batch = d_num > 0 ? X_num_d[:, idx] : _to_device(zeros(Float32, 0, bs), dev)

            if d_num > 0
                x_num_noised, ε = _gaussian_forward(x_num_batch, alphas_cumprod_d, t_batch, rng, dev)
            else
                x_num_noised = _to_device(zeros(Float32, 0, bs), dev)
                ε = _to_device(zeros(Float32, 0, bs), dev)
            end

            if plan !== nothing
                log_x_cat_start = log_x_cat_all[:, idx]
                batch_coef = _batch_coefs(sched_d, _to_device(t_batch, dev))
                log_x_cat_t = _multinomial_q_sample(log_x_cat_start, plan, batch_coef,
                                                    log_K, rng, dev)
            else
                log_x_cat_start = _to_device(zeros(Float32, 0, bs), dev)
                log_x_cat_t     = log_x_cat_start
            end

            # ── Per-sample gradient clipping ────────────────────────────
            gs_sum = _grad_zero(ps_all)
            batch_loss = 0.0

            for si in 1:bs
                xn_i  = d_num > 0 ? x_num_noised[:, si:si] : _to_device(zeros(Float32, 0, 1), dev)
                xc_i  = size(log_x_cat_t, 1) > 0 ? log_x_cat_t[:, si:si] : _to_device(zeros(Float32, 0, 1), dev)
                xc_orig_i = size(log_x_cat_start, 1) > 0 ? log_x_cat_start[:, si:si] : _to_device(zeros(Float32, 0, 1), dev)
                t_i_d = _to_device(Float32.([t_batch[si] - 1]), dev)
                ε_i   = d_num > 0 ? ε[:, si:si] : _to_device(zeros(Float32, 0, 1), dev)
                y_i   = conditional ? _to_device(y_indices[idx[si]:idx[si]], dev) : nothing
                coef_i = _scalar_coefs(sched, t_batch[si])

                l, _, g = _compute_grad(AD_BACKEND, ps_all) do p
                    _diffusion_loss(backbone, emb_layer,
                                    p.backbone, p.emb,
                                    st_bb, st_emb,
                                    xn_i, xc_i, xc_orig_i,
                                    t_i_d, coef_i, ε_i, d_num, plan,
                                    T, log_K, lca_T, l1m_T, y_i)
                end
                batch_loss += l
                gnorm = sqrt(_grad_sqnorm(g))
                clip_factor = min(1.0, C / max(gnorm, 1e-12))
                g_clipped = _grad_scale(g, clip_factor)
                gs_sum = _grad_add(gs_sum, g_clipped)
            end

            epoch_loss += batch_loss / bs
            n_batches  += 1

            # Normalize by the *expected* lot size (not the realized one) and
            # add Gaussian noise calibrated to the clip norm C.
            gs_avg = _grad_scale(gs_sum, 1.0 / expected_bs)
            noise_scale = σ_noise * C / expected_bs
            gs_noisy = _grad_add_noise!(gs_avg, noise_scale, rng)

            opt_state, ps_all = Optimisers.update(opt_state, ps_all, gs_noisy)
        end

        avg_loss = epoch_loss / max(n_batches, 1)
        _check_finite_loss(avg_loss, epoch, epochs, cur_lr)

        # Progress report
        if epoch == 1 || epoch % report_every == 0 || epoch == epochs
            elapsed  = time() - t_start
            eta      = elapsed / epoch * (epochs - epoch)
            @info "DP-SGD Epoch $(epoch)/$(epochs)  loss=$(round(avg_loss; digits=4))  lr=$(round(cur_lr; sigdigits=3))  elapsed=$(round(Int, elapsed))s  ETA=$(round(Int, eta))s"
        end
    end

    return ps_all.backbone, ps_all.emb, st_bb, st_emb
end

# ═══════════════════════════════════════════════════════════════════════════
# 11. Reverse Sampling (Denoising)
# ═══════════════════════════════════════════════════════════════════════════

"""
Draw `n` rows, discarding any whose numeric block came back non-finite.

The ε-parametrization divides by `√ᾱ`, which at the noisiest timesteps is
around 2000 for a 100-step cosine schedule.  That amplifies the model's noise
prediction error by the same factor, so an undertrained model can drive the
reverse process to overflow and produce `Inf` or `NaN` rows.  The reference
implementation expects this and filters such rows before returning; this does
the same, redrawing to make up the shortfall.

Persistent failure means the model itself is unusable rather than unlucky, so
it raises with the likely cause instead of returning quietly corrupted data.
"""
function _denoise_sample_finite(backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
                                sched, d_num, cat_dims, n, rng, y_idx;
                                max_attempts::Int = 8)
    d_cat_total = sum(cat_dims; init = 0)
    num_parts = Matrix{Float32}[]
    cat_parts = Matrix{Float32}[]
    y_parts   = Vector{Int}[]
    have = 0

    for attempt in 1:max_attempts
        want = n - have
        want <= 0 && break
        # Over-draw a little after the first pass, so a low yield does not
        # need many rounds to converge.
        draw = attempt == 1 ? want : min(2 * want, 4 * n)

        y_draw = y_idx === nothing ? nothing : y_idx[1:min(draw, length(y_idx))]
        if y_idx !== nothing && draw > length(y_idx)
            y_draw = vcat(y_idx, y_idx[1:(draw - length(y_idx))])
        end

        xn, xc = _denoise_sample(backbone, emb_layer, ps_bb, ps_emb,
                                 st_bb, st_emb, sched, d_num, cat_dims,
                                 draw, rng, y_draw)
        xn_h = Array(xn); xc_h = Array(xc)

        keep = d_num > 0 ?
            [all(isfinite, view(xn_h, :, j)) for j in 1:size(xn_h, 2)] :
            trues(size(xc_h, 2))
        nkeep = count(keep)
        if nkeep > 0
            take = min(nkeep, n - have)
            idx  = findall(keep)[1:take]
            push!(num_parts, d_num > 0 ? xn_h[:, idx] : zeros(Float32, 0, take))
            push!(cat_parts, d_cat_total > 0 ? xc_h[:, idx] : zeros(Float32, 0, take))
            y_draw !== nothing && push!(y_parts, y_draw[idx])
            have += take
        end

        if attempt == 1 && nkeep < draw
            @warn "Discarded $(draw - nkeep) of $draw sampled rows with " *
                  "non-finite values and redrew them. This usually means the " *
                  "diffusion model is undertrained — try more epochs, a lower " *
                  "learning rate, or fewer timesteps."
        end
    end

    if have < n
        error("DiffusionGenerator could not draw $n finite rows " *
              "($have after $max_attempts attempts). The reverse process is " *
              "diverging, which points at an undertrained or unstable model: " *
              "try more epochs, a lower learning rate, or fewer timesteps.")
    end

    x_num = d_num > 0 ? reduce(hcat, num_parts) : zeros(Float32, 0, n)
    x_cat = d_cat_total > 0 ? reduce(hcat, cat_parts) : zeros(Float32, 0, n)
    y_out = isempty(y_parts) ? nothing : reduce(vcat, y_parts)
    return x_num, x_cat, y_out
end

"""
Sample `n` rows from a trained TabDDPM model via reverse denoising.

Follows `GaussianMultinomialDiffusion.sample` in the reference implementation:
the full `T`-step DDPM reverse process, with the Gaussian branch using the
`eps` parametrization and the posterior mean/variance, and the categorical
branch stepping through the multinomial posterior in log space.

`y_idx` supplies the class label per row for conditional models, or `nothing`.
"""
function _denoise_sample(backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
                         sched, d_num, cat_dims, n, rng, y_idx)
    T = sched.T
    d_cat_total = sum(cat_dims; init = 0)

    st_bb = Lux.testmode(st_bb)
    gdev, cdev = _get_devices()
    log_K = _to_device(_log_K_vector(cat_dims), gdev)
    plan  = _block_plan(cat_dims, gdev)

    # Reverse-process variance: [posterior_variance[2], betas[2:T]]
    model_variance = vcat(sched.posterior_variance[2], sched.betas[2:end])
    model_log_variance = log.(model_variance)
    sqrt_recip_ac   = sqrt.(1f0 ./ sched.alphas_cumprod)
    sqrt_recipm1_ac = sqrt.(1f0 ./ sched.alphas_cumprod .- 1f0)

    y_dev = y_idx === nothing ? nothing : _to_device(y_idx, gdev)

    # Start from pure noise / a uniform categorical draw
    x_num = d_num > 0 ? _to_device(randn(rng, Float32, d_num, n), gdev) :
                        _to_device(zeros(Float32, 0, n), gdev)
    log_x_cat = if d_cat_total > 0
        _log_sample_categorical(_to_device(zeros(Float32, d_cat_total, n), gdev),
                                plan, rng, gdev)
    else
        _to_device(zeros(Float32, 0, n), gdev)
    end

    for t in T:-1:1
        coef = _scalar_coefs(sched, t)
        # Timestep embedding uses 0-based t, as in the reference code
        t_batch = _to_device(fill(Float32(t - 1), n), gdev)
        t_emb, _ = Lux.apply(emb_layer, t_batch, ps_emb, st_emb)

        features = if d_num > 0 && d_cat_total > 0
            vcat(x_num, log_x_cat)
        elseif d_num > 0
            x_num
        else
            log_x_cat
        end

        output, _ = Lux.apply(backbone, (features, t_emb, y_dev), ps_bb, st_bb)

        # ── Gaussian reverse step ──────────────────────────────────────
        if d_num > 0
            ε_pred = output[1:d_num, :]

            # x̂₀ from the ε parametrization
            x0_pred = sqrt_recip_ac[t] .* x_num .- sqrt_recipm1_ac[t] .* ε_pred

            # Posterior mean q(x_{t-1} | x_t, x̂₀)
            post_mean = sched.posterior_mean_coef1[t] .* x0_pred .+
                        sched.posterior_mean_coef2[t] .* x_num

            if t > 1   # no noise at the final step
                z = _to_device(randn(rng, Float32, d_num, n), gdev)
                x_num = post_mean .+ exp(0.5f0 * model_log_variance[t]) .* z
            else
                x_num = post_mean
            end
        end

        # ── Multinomial reverse step ───────────────────────────────────
        if d_cat_total > 0
            model_out_cat = output[(d_num + 1):end, :]
            log_model_prob = _p_pred(model_out_cat, log_x_cat, plan, coef, log_K)
            log_x_cat = _log_sample_categorical(log_model_prob, plan, rng, gdev)
        end
    end

    # Categoricals come back as one-hot for the caller's argmax decoding
    x_cat = d_cat_total > 0 ? exp.(log_x_cat) : log_x_cat
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
    if gen.target !== nothing && !(gen.target in col_names)
        throw(ArgumentError("target column :$(gen.target) not found in the table"))
    end

    X_num, X_cat_oh, cat_indices, info = _preprocess(
        cols, col_names, col_kinds, id_set, hints, nrows; target = gen.target)

    d_num       = info.d_num
    d_cat_total = info.d_cat_total
    cat_dims_v  = info.cat_dims

    if d_num == 0 && d_cat_total == 0
        error("No statistical columns for DiffusionGenerator after preprocessing.")
    end

    # ── Noise schedule (cosine, the TabDDPM default) ───────────────────
    n_steps = gen.num_timesteps
    betas, alphas_cumprod = _cosine_schedule(n_steps)
    sched = _schedule_constants(betas)

    # ── Build model ────────────────────────────────────────────────────
    embed_dim = gen.embed_dim
    d_in      = d_num + d_cat_total   # timestep handled via addition, not concat
    n_classes = length(info.target_levels)
    layer_widths = if !isempty(gen.d_layers)
        gen.d_layers
    else
        hidden = gen.hidden_dim > 0 ? gen.hidden_dim :
                 min(256, max(64, 4 * (d_num + d_cat_total)))
        fill(hidden, gen.n_blocks)
    end

    backbone, _ = _build_model(d_in, d_num, cat_dims_v;
                               d_layers = layer_widths,
                               embed_dim = embed_dim, dropout = gen.dropout,
                               n_classes = n_classes)
    emb_layer = SinusoidalEmbedding(embed_dim)

    lux_rng = Random.MersenneTwister(42)  # deterministic init
    ps_bb, st_bb   = Lux.setup(lux_rng, backbone)
    ps_emb, st_emb = Lux.setup(lux_rng, emb_layer)

    # ── Detect device (GPU if available) ─────────────────────────────
    gdev, cdev = _get_devices()
    @info "DiffusionGenerator: training on $(gdev)" *
          (n_classes > 0 ? " (class-conditional on :$(gen.target), $n_classes classes)" : "")

    # ── Train ──────────────────────────────────────────────────────────
    if gen.dp
        ps_bb, ps_emb, st_bb, st_emb = _train_dpsgd!(
            backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
            X_num, X_cat_oh, info.y_indices,
            sched, cat_dims_v, d_num,
            gen.epochs, gen.batch_size, gen.lr, gen.lr_warmup,
            gen.weight_decay, privacy, rng, gdev)
    else
        ps_bb, ps_emb, st_bb, st_emb = _train_standard!(
            backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
            X_num, X_cat_oh, info.y_indices,
            sched, cat_dims_v, d_num,
            gen.epochs, gen.batch_size, gen.lr, gen.lr_warmup,
            gen.weight_decay, gen.ema_decay, rng, gdev)
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
        info.num_references,
        info.cat_levels, info.cat_dims,
        info.num_round,
        gen.target, info.target_levels, info.class_dist,
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
    sample(model::FittedDiffusionModel, n; rng)

Generate `n` synthetic rows by running the full reverse diffusion process
[Kotelnikov et al. 2023].  For a class-conditional model, labels are first
drawn from the empirical class distribution of the training data and the
denoiser is conditioned on them.
"""
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

    d_num     = length(model.num_columns)
    cat_dims  = model.cat_dims
    sched     = _schedule_constants(model.betas)

    # ── Move model to best available device for sampling ───────────────
    gdev, cdev = _get_devices()
    ps_bb  = _to_device(full_ps.backbone, gdev)
    ps_emb = _to_device(full_ps.emb, gdev)
    st_bb  = _to_device(full_st.backbone, gdev)
    st_emb = _to_device(full_st.emb, gdev)

    # ── Draw class labels from the empirical distribution ──────────────
    y_idx = if model.target === nothing || isempty(model.class_dist)
        nothing
    else
        cdf = cumsum(model.class_dist)
        [searchsortedfirst(cdf, rand(rng)) for _ in 1:n]
    end

    # ── Reverse denoise (rejecting any non-finite rows) ────────────────
    x_num, x_cat, y_kept = _denoise_sample_finite(
        backbone, emb_layer, ps_bb, ps_emb, st_bb, st_emb,
        sched, d_num, cat_dims, n, rng, y_idx)
    y_kept !== nothing && (y_idx = y_kept)

    # Move results back to CPU for post-processing
    x_num = _to_device(x_num, cdev)
    x_cat = _to_device(x_cat, cdev)

    # ── Unpack into result dict ────────────────────────────────────────
    result = Dict{Symbol, Vector}()

    # Numeric columns: inverse Gaussian quantile transform
    for (j, name) in enumerate(model.num_columns)
        ref = model.num_references[j]
        z_vals = x_num[j, :]
        raw = [_quantile_inverse(Float32(z), ref) for z in z_vals]

        idx = findfirst(==(name), model.column_names)
        kind = model.column_kinds[idx]
        if kind == :integer || model.num_round[j]
            result[name] = round.(Int64, raw)
        else
            result[name] = Float64.(raw)
        end
    end

    # Target column: the labels the model was conditioned on
    if model.target !== nothing && y_idx !== nothing
        lvls = model.target_levels
        result[model.target] = [lvls[k] for k in y_idx]
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
