# ─── Display ────────────────────────────────────────────────────────────────
#
# Fitted models hold the whole learned artifact — sorted empirical marginals,
# copula internals, noisy marginal tables, neural network parameters.  The
# default struct display dumps all of it: a two-column, 200-row copula model
# prints over 4,000 characters, and a diffusion model holds millions of
# weights.  These methods show what a caller actually wants to know.

"""Count modelled vs identifier columns for the summary line."""
function _column_summary(m::AbstractFittedModel)
    n_id  = length(m.identifier_columns)
    n_all = length(m.column_names)
    return n_all, n_all - n_id, n_id
end

function _summary_line(io::IO, m::AbstractFittedModel, label::AbstractString)
    n_all, n_mod, n_id = _column_summary(m)
    print(io, label, ": ", n_all, " column", n_all == 1 ? "" : "s")
    print(io, " (", n_mod, " modelled")
    n_id > 0 && print(io, ", ", n_id, " identifier", n_id == 1 ? "" : "s")
    print(io, "), fitted on ", m.n_original, " rows")
end

# Compact form, used inside containers.
Base.show(io::IO, m::FittedCopulaModel)   = _summary_line(io, m, "FittedCopulaModel")
Base.show(io::IO, m::FittedMSTModel)      = _summary_line(io, m, "FittedMSTModel")
Base.show(io::IO, m::FittedDPCopulaModel) = _summary_line(io, m, "FittedDPCopulaModel")

"""
Show the guarantee a fitted model carries, when it carries one.

A private model that displays like a public one invites exactly the mistake
this line prevents: treating a synthetic table as protected without checking
what it was protected with.
"""
function _privacy_line(io::IO, m)
    b = privacy_budget(m)
    b === nothing && return nothing
    print(io, "
  privacy: ε = ", b.epsilon, ", δ = ", b.delta)
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::FittedCopulaModel)
    _summary_line(io, m, "FittedCopulaModel")
    if isnothing(m.copula)
        print(io, "\n  copula:  none (columns sampled independently)")
    else
        print(io, "\n  copula:  ", nameof(typeof(m.copula)),
                  " over ", length(m.copula_columns), " columns")
    end
    nmiss = count(>(0.0), values(m.missingness))
    nmiss > 0 && print(io, "\n  missing: ", nmiss, " column(s) carry missingness")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::FittedMSTModel)
    _summary_line(io, m, "FittedMSTModel")
    _privacy_line(io, m)
    print(io, "\n  tree:    ", length(m.tree_edges), " edge",
              length(m.tree_edges) == 1 ? "" : "s",
              ", root = :", m.stat_columns[m.root])
    print(io, "\n  bins:    ", sum(d.n_bins for d in values(m.discretization);
                                   init = 0), " total across ",
              length(m.discretization), " columns")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::FittedDPCopulaModel)
    _summary_line(io, m, "FittedDPCopulaModel")
    _privacy_line(io, m)
    if isnothing(m.copula)
        print(io, "\n  copula:  none (columns sampled independently)")
    else
        print(io, "\n  copula:  private Gaussian over ",
                  length(m.copula_columns), " columns")
    end
    return nothing
end

# FittedDiffusionModel is constructed by the Lux extension, but the type is
# defined here, so its display belongs here too.
Base.show(io::IO, m::FittedDiffusionModel) =
    _summary_line(io, m, "FittedDiffusionModel")

function Base.show(io::IO, ::MIME"text/plain", m::FittedDiffusionModel)
    _summary_line(io, m, "FittedDiffusionModel")
    _privacy_line(io, m)
    print(io, "\n  features: ", length(m.num_columns), " numeric, ",
              length(m.cat_columns), " categorical")
    print(io, "\n  schedule: ", m.n_steps, " diffusion timesteps")
    if m.target === nothing
        print(io, "\n  sampling: unconditional")
    else
        print(io, "\n  sampling: conditional on :", m.target,
                  " (", length(m.target_levels), " classes)")
    end
    return nothing
end
