# ─── Abstract Hierarchy ──────────────────────────────────────────────────────

abstract type AbstractGenerator end
abstract type AbstractPublicGenerator  <: AbstractGenerator end
abstract type AbstractPrivateGenerator <: AbstractGenerator end

abstract type AbstractFittedModel end

# ─── Privacy Budget ──────────────────────────────────────────────────────────

"""
    PrivacyBudget(; epsilon, delta=1e-5)

Differential privacy parameters used by private generators
(`MSTGenerator`, `DPCopulaGenerator`, `DiffusionGenerator(dp=true)`).
"""
Base.@kwdef struct PrivacyBudget
    epsilon::Float64
    delta::Float64 = 1e-5

    function PrivacyBudget(epsilon, delta)
        epsilon > 0   || throw(ArgumentError("ε must be positive, got $epsilon"))
        0 ≤ delta < 1 || throw(ArgumentError("δ must be in [0, 1), got $delta"))
        new(Float64(epsilon), Float64(delta))
    end
end

# ─── Generator Configs ───────────────────────────────────────────────────────

"""
    CopulaGenerator(copula_type::Symbol=:beta)

Public (non-private) copula-based synthetic data generator.
`copula_type` must be `:beta` or `:gaussian`.
"""
struct CopulaGenerator <: AbstractPublicGenerator
    copula_type::Symbol

    function CopulaGenerator(copula_type::Symbol)
        copula_type in (:beta, :gaussian) ||
            throw(ArgumentError("copula_type must be :beta or :gaussian, got :$copula_type"))
        new(copula_type)
    end
end
CopulaGenerator() = CopulaGenerator(:beta)

"""
    MSTGenerator()

Private synthetic data via MST (McKenna et al. 2021): measure all 1-way
marginals → select a spanning tree with the exponential mechanism → measure the
selected 2-way marginals → reconcile every measurement with Private-PGM
(McKenna et al. 2019) → sample ancestrally from the resulting conditionals.
Satisfies (ε,δ)-DP via zCDP composition; the reconciliation step is
post-processing and costs no budget.

Marginal order is not a parameter. MST *is* the spanning tree over 2-way
marginals: the tree is what makes belief propagation exact, and 3-way marginals
would need junction-tree inference and a different budget argument. The
published mechanism for adaptively chosen higher-order marginals is AIM
(McKenna et al. 2022), which is a different algorithm rather than a setting on
this one.

Domain compression — folding bins whose noisy count falls below 3σ into one
"other" category before selection — is included, and was measured rather than
adopted on the paper's authority: better or neutral on three of four real
tables, at a small cost on one whose columns are mostly binary. See the MST
implementation note in REQUIREMENTS.md.
"""
struct MSTGenerator <: AbstractPrivateGenerator end

"""
    DPCopulaGenerator()

DP-noisy histogram marginals + private covariance Gaussian copula.
Suited for continuous-heavy tables under moderate ε.
"""
struct DPCopulaGenerator <: AbstractPrivateGenerator end

"""
    DiffusionGenerator(; dp=false, epochs=100, batch_size=512, target=nothing)

TabDDPM [Kotelnikov et al. 2023] with optional DP-SGD.  Requires the `LuxExt`
package extension (`using Lux, Zygote` before calling `fit`).

Architecture and training follow the reference implementation: a plain MLP
denoiser (`Linear → ReLU → Dropout` blocks) with additive sinusoidal timestep
conditioning, a cosine β schedule, Gaussian diffusion on numeric features and
multinomial diffusion on categoricals, AdamW with linear learning-rate
annealing, and an exponential moving average of the denoiser weights used for
sampling.

# Keyword arguments
- `target`: name of the label column for class-conditional generation
  (the paper's `is_y_cond=true`).  When set, the denoiser is conditioned on an
  embedding of the label and sampling first draws labels from the empirical
  class distribution.  When `nothing`, the model is unconditional and the label,
  if any, is modelled as an ordinary categorical column.
- `hidden_dim`: width of each MLP block (0 = auto).
- `n_blocks`: number of MLP blocks.
- `d_layers`: explicit per-layer widths, e.g. `[256, 1024, 1024, 256]`.  When
  non-empty this overrides `hidden_dim`/`n_blocks`, matching the paper's
  per-dataset tuned architectures.
- `num_timesteps`: length of the diffusion process (the paper tunes this;
  its Adult configuration uses 100).
- `embed_dim`: timestep-embedding width (the paper's `dim_t`).
- `lr`, `weight_decay`: AdamW parameters.
- `lr_warmup`: linear warmup epochs prepended to the annealing schedule
  (0 = the paper's plain linear anneal).
- `ema_decay`: EMA rate for the sampling weights (0 disables EMA).
"""
Base.@kwdef struct DiffusionGenerator <: AbstractGenerator
    dp::Bool           = false
    epochs::Int        = 100
    batch_size::Int    = 512
    hidden_dim::Int    = 0      # 0 = auto: min(256, max(64, 4·d_features))
    n_blocks::Int      = 4
    d_layers::Vector{Int} = Int[]   # explicit widths; overrides hidden_dim/n_blocks
    num_timesteps::Int = 1000
    embed_dim::Int     = 128    # dim_t in the reference implementation
    dropout::Float64   = 0.0
    lr::Float64        = 1e-3
    lr_warmup::Int     = 0      # linear warmup epochs (0 = no warmup)
    weight_decay::Float64 = 1e-4
    ema_decay::Float64 = 0.999  # 0 = disable EMA
    target::Union{Symbol, Nothing} = nothing   # class-conditional label column

    function DiffusionGenerator(dp, epochs, batch_size, hidden_dim, n_blocks,
                                d_layers, num_timesteps, embed_dim, dropout, lr,
                                lr_warmup, weight_decay, ema_decay, target)
        epochs > 0     || throw(ArgumentError("epochs must be positive, got $epochs"))
        batch_size > 0 || throw(ArgumentError("batch_size must be positive, got $batch_size"))
        hidden_dim >= 0 || throw(ArgumentError("hidden_dim must be non-negative, got $hidden_dim"))
        n_blocks > 0   || throw(ArgumentError("n_blocks must be positive, got $n_blocks"))
        all(>(0), d_layers) || throw(ArgumentError("d_layers must be all positive, got $d_layers"))
        num_timesteps > 0 || throw(ArgumentError("num_timesteps must be positive, got $num_timesteps"))
        embed_dim > 0  || throw(ArgumentError("embed_dim must be positive, got $embed_dim"))
        0.0 <= dropout < 1.0 || throw(ArgumentError("dropout must be in [0,1), got $dropout"))
        lr > 0         || throw(ArgumentError("lr must be positive, got $lr"))
        lr_warmup >= 0 || throw(ArgumentError("lr_warmup must be non-negative, got $lr_warmup"))
        weight_decay >= 0 || throw(ArgumentError("weight_decay must be non-negative, got $weight_decay"))
        0.0 <= ema_decay < 1.0 || throw(ArgumentError("ema_decay must be in [0,1), got $ema_decay"))
        new(dp, epochs, batch_size, hidden_dim, n_blocks, d_layers, num_timesteps,
            embed_dim, dropout, lr, lr_warmup, weight_decay, ema_decay, target)
    end
end

# ─── Column Schema Hints ────────────────────────────────────────────────────

const VALID_COLUMN_KINDS = (:continuous, :integer, :categorical, :binary, :constant, :identifier)

"""
    ColumnHint(; name, kind, levels=nothing)

Override auto-detected column type. `kind` must be one of:
`:continuous`, `:integer`, `:categorical`, `:binary`, `:constant`, `:identifier`.
"""
Base.@kwdef struct ColumnHint
    name::Symbol
    kind::Symbol
    levels::Union{Nothing, Vector} = nothing

    function ColumnHint(name, kind, levels)
        kind in VALID_COLUMN_KINDS ||
            throw(ArgumentError("ColumnHint kind must be one of $VALID_COLUMN_KINDS, got :$kind"))
        new(name, kind, levels)
    end
end

# ─── Marginal Types ─────────────────────────────────────────────────────────

struct EmpiricalMarginal
    sorted_values::Vector{Float64}
    original_eltype::Type
end

struct CategoricalMarginal
    levels::Vector
    probs::Vector{Float64}
end

struct ConstantMarginal{T}
    value::T
end

# ─── Shared type aliases ────────────────────────────────────────────────────

"""Union of all marginal types — used in marginals dicts."""
const Marginal = Union{EmpiricalMarginal, CategoricalMarginal, ConstantMarginal}

"""Union of valid fill specs for identifier columns."""
const FillSpec = Union{Symbol, String, Function}

# ─── Fitted Model Types ─────────────────────────────────────────────────────

"""
    FittedCopulaModel <: AbstractFittedModel

Result of fitting a `CopulaGenerator`. Contains all information needed
to sample synthetic data.
"""
struct FittedCopulaModel{C, M} <: AbstractFittedModel
    column_names::Vector{Symbol}       # all columns in original order
    column_kinds::Vector{Symbol}       # :continuous, :integer, etc. or :identifier
    marginals::Dict{Symbol, Marginal}  # stat columns only
    missingness::Dict{Symbol, Float64} # stat columns only
    copula::C                          # fitted copula or nothing
    copula_columns::Vector{Symbol}     # numeric stat columns used in copula
    n_original::Int
    identifier_columns::Vector{Symbol}
    identifier_fills::Dict{Symbol, FillSpec}
    materializer::M                    # Tables.materializer of original input
    rng::AbstractRNG
end

# ─── Discretization Info (MST engine) ──────────────────────────────────────

"""
Metadata for discretizing a column into integer bin indices.

- Continuous/integer: `bin_edges` is a `k+1`-length vector of breakpoints.
- Categorical/binary/constant: `levels` maps bin index → original value.
"""
struct DiscretizationInfo
    kind::Symbol
    original_eltype::Type
    bin_edges::Union{Nothing, Vector{Float64}}  # for continuous / integer
    levels::Union{Nothing, Vector}              # for categorical / binary / constant
    n_bins::Int
end

# ─── DP-specific marginal types ────────────────────────────────────────────

"""
A DP-noisy histogram marginal for continuous / integer columns.

Stores `k` bin probabilities and `k+1` bin edges.  Sampling draws a bin
from the probability vector and then samples uniformly within the bin.
"""
struct DPHistogramMarginal
    bin_edges::Vector{Float64}
    probs::Vector{Float64}
    original_eltype::Type
end

# ─── Fitted Model Types (Phase 2) ─────────────────────────────────────────

"""
    FittedMSTModel <: AbstractFittedModel

Result of fitting an `MSTGenerator`.  Stores discretization metadata, a
spanning-tree structure, and noisy conditional distributions for sampling.
"""
struct FittedMSTModel{M} <: AbstractFittedModel
    column_names::Vector{Symbol}
    column_kinds::Vector{Symbol}
    stat_columns::Vector{Symbol}
    discretization::Dict{Symbol, DiscretizationInfo}
    tree_edges::Vector{Tuple{Int, Int}}            # (parent, child) indices into stat_columns
    root::Int                                       # index into stat_columns
    root_marginal::Vector{Float64}                  # probability vector for root column
    conditionals::Dict{Tuple{Int,Int}, Matrix{Float64}}  # P(child | parent)
    missingness::Dict{Symbol, Float64}
    n_original::Int
    identifier_columns::Vector{Symbol}
    identifier_fills::Dict{Symbol, FillSpec}
    materializer::M
    rng::AbstractRNG
end

# Union of marginal types used by DP generators.
const DPMarginal = Union{DPHistogramMarginal, CategoricalMarginal, ConstantMarginal}

"""
    FittedDPCopulaModel <: AbstractFittedModel

Result of fitting a `DPCopulaGenerator`.  DP-noisy histogram marginals
with an optional private-covariance Gaussian copula.
"""
struct FittedDPCopulaModel{C, M} <: AbstractFittedModel
    column_names::Vector{Symbol}
    column_kinds::Vector{Symbol}
    marginals::Dict{Symbol, DPMarginal}
    missingness::Dict{Symbol, Float64}
    copula::C                          # GaussianCopula or nothing
    copula_columns::Vector{Symbol}
    n_original::Int
    identifier_columns::Vector{Symbol}
    identifier_fills::Dict{Symbol, FillSpec}
    materializer::M
    rng::AbstractRNG
end

# ─── Fitted Model Types (Phase 3) ─────────────────────────────────────────

"""
    FittedDiffusionModel <: AbstractFittedModel

Result of fitting a `DiffusionGenerator`.  Stores the trained Lux model
(TabDDPM), preprocessing metadata, and diffusion schedule.
"""
struct FittedDiffusionModel{L, P, S, Mat} <: AbstractFittedModel
    column_names::Vector{Symbol}
    column_kinds::Vector{Symbol}
    # Preprocessing metadata
    num_columns::Vector{Symbol}          # numeric columns in model order
    cat_columns::Vector{Symbol}          # categorical columns in model order
    num_references::Vector{Vector{Float32}}  # sorted training values per numeric column (quantile transform)
    cat_levels::Dict{Symbol, Vector}     # column → sorted levels
    cat_dims::Vector{Int}                # one-hot width per cat column
    num_round::Vector{Bool}              # round to integers on inverse transform
    # Class conditioning (nothing = unconditional)
    target::Union{Symbol, Nothing}       # label column, or nothing
    target_levels::Vector                # label values in model order
    class_dist::Vector{Float64}          # empirical class distribution
    # Trained neural network (Lux)
    lux_model::L                         # Lux model (backbone + embedding)
    trained_params::P                    # EMA params when EMA enabled, else raw
    model_state::S                       # Lux state
    # Diffusion schedule
    n_steps::Int
    betas::Vector{Float32}
    alphas_cumprod::Vector{Float32}
    # Standard fields
    missingness::Dict{Symbol, Float64}
    n_original::Int
    identifier_columns::Vector{Symbol}
    identifier_fills::Dict{Symbol, FillSpec}
    materializer::Mat
    rng::AbstractRNG
end
