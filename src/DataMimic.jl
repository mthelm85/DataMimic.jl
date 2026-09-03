module DataMimic

using Tables
using Random
import Serialization
import StatsBase
import Copulas
import LinearAlgebra
import Optimisers
import EvoTrees
import SpecialFunctions

include("types.jl")
include("detect.jl")
include("identifiers.jl")
include("privacy.jl")
include("engines/copula.jl")
include("engines/mst.jl")
include("engines/dp_copula.jl")
include("fit.jl")
include("sample.jl")
include("serialize.jl")
include("show.jl")
include("evaluate/Evaluate.jl")

# Abstract types
export AbstractGenerator, AbstractPublicGenerator, AbstractPrivateGenerator
export AbstractFittedModel

# Generator configs
export CopulaGenerator
export MSTGenerator, DPCopulaGenerator, DiffusionGenerator

# Other types
export PrivacyBudget, ColumnHint
export FittedCopulaModel, FittedMSTModel, FittedDPCopulaModel, FittedDiffusionModel

# Functions
export fit, sample, synthesize
export save, load

# Evaluation submodule
using .Evaluate
export Evaluate
export privacy_budget
export fidelity_score, privacy_dcr, utility_tstr
export jensen_shannon, pairwise_marginal_error, privacy_utility_sweep
export compare

end
