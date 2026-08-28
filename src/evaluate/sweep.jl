# ─── privacy_utility_sweep ─────────────────────────────────────────────────
#
# REQ-EVL-014: Fit and sample at each ε, evaluate with metric_fn
# REQ-EVL-015: Accept any metric_fn(real, synth) returning a NamedTuple

"""
    privacy_utility_sweep(generator, table, epsilons, metric_fn;
                          n = nothing, hints = ColumnHint[],
                          identifiers = Symbol[],
                          fill = Dict{Symbol, FillSpec}(),
                          rng = Random.default_rng(),
                          delta = 1e-5,
                          metric_kw...) -> Vector{NamedTuple}

Run a privacy–utility curve: for each ε in `epsilons`, fit `generator`
with `PrivacyBudget(; epsilon=ε, delta)`, sample `n` rows (default =
number of rows in `table`), and evaluate with `metric_fn(real, synth)`.

`generator` must be an `AbstractPrivateGenerator` or a
`DiffusionGenerator(dp=true)`.

Returns a vector of `(; epsilon, delta, metric_result)` named tuples,
one per ε value, sorted by ascending ε.

Standard usage:
```julia
results = privacy_utility_sweep(
    MSTGenerator(), data, [0.1, 0.5, 1.0, 5.0, 10.0],
    fidelity_score
)
```
"""
function privacy_utility_sweep(generator, table, epsilons::AbstractVector,
                                metric_fn;
                                n::Union{Nothing, Int} = nothing,
                                hints::Vector{ColumnHint} = ColumnHint[],
                                identifiers::Vector{Symbol} = Symbol[],
                                fill = Dict{Symbol, FillSpec}(),
                                rng::AbstractRNG = Random.default_rng(),
                                delta::Float64 = 1e-5,
                                metric_kw...)
    Tables.istable(table) || throw(ArgumentError(
        "table must be a Tables.jl table"))
    isempty(epsilons) && throw(ArgumentError(
        "epsilons must be non-empty"))
    all(e -> e > 0, epsilons) || throw(ArgumentError(
        "all epsilon values must be positive"))

    # Determine sample size
    cols = Tables.columns(table)
    col_names = collect(Symbol, Tables.columnnames(cols))
    n_rows = length(Tables.getcolumn(cols, first(col_names)))
    sample_n = isnothing(n) ? n_rows : n

    sorted_eps = sort(collect(Float64, epsilons))

    results = NamedTuple[]

    for ε in sorted_eps
        budget = PrivacyBudget(; epsilon = ε, delta = delta)

        model = DataMimic.fit(generator, table;
                              privacy = budget,
                              hints = hints,
                              identifiers = identifiers,
                              fill = fill,
                              rng = copy(rng))

        synth = DataMimic.sample(model, sample_n)

        metric_result = metric_fn(table, synth; metric_kw...)

        push!(results, (;
            epsilon       = ε,
            delta         = delta,
            metric_result = metric_result,
        ))
    end

    return results
end
