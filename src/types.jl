# Marginal types

struct EmpiricalMarginal
    sorted_values::Vector       # sorted non-missing observed values
    original_eltype::Type       # eltype of the original column (non-missing part)
end

struct CategoricalMarginal
    levels::Vector              # unique observed levels
    probs::Vector{Float64}      # empirical probability per level
end

struct ConstantMarginal
    value::Any                  # the single observed value (may be missing)
end

# Top-level model struct
struct SynthModel
    column_names::Vector{Symbol}
    column_types::Vector{Symbol}  # :continuous, :integer, :categorical, :binary, :constant
    marginals::Dict{Symbol, Any}
    missingness::Dict{Symbol, Float64}
    copula::Any                   # fitted BetaCopula, or Nothing
    copula_columns::Vector{Symbol}
    nrows_original::Int
    scrambled::Vector{Symbol}     # columns whose sampled values are character-scrambled
end
