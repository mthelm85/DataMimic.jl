# ─── DataMimic.Evaluate ─────────────────────────────────────────────────────
#
# Submodule providing six standard evaluation metrics for synthetic data:
#
#   fidelity_score(real, synth)           — per-column KS / TVD + correlation
#   privacy_dcr(real, synth)              — Distance to Closest Record
#   utility_tstr(real, synth, target)     — Train on Synthetic, Test on Real
#   jensen_shannon(real, synth)           — per-column Jensen–Shannon divergence
#   pairwise_marginal_error(real, synth)  — 2-way/3-way joint distribution TVD
#   privacy_utility_sweep(gen, table, εs, metric_fn) — ε sweep
#
# References:
#   [Zhao et al. 2021]    — DCR metric
#   [Esteban et al. 2017] — TSTR protocol
#   [Lin 1991]            — Jensen–Shannon divergence
#   [McKenna et al. 2019] — Pairwise marginal error

module Evaluate

import Tables
import StatsBase
import LinearAlgebra
import DecisionTree
import Random
import Random: AbstractRNG

using ..DataMimic: DataMimic, PrivacyBudget, ColumnHint, FillSpec

include("fidelity.jl")
include("dcr.jl")
include("tstr.jl")
include("jsd.jl")
include("marginal_error.jl")
include("sweep.jl")

export fidelity_score, privacy_dcr, utility_tstr
export jensen_shannon, pairwise_marginal_error, privacy_utility_sweep

end
