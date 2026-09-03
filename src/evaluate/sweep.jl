# ─── privacy_utility_sweep ─────────────────────────────────────────────────
#
# REQ-EVL-014: Fit and sample at each ε, evaluate with metric_fn
# REQ-EVL-015: Accept any metric_fn(real, synth) returning a NamedTuple

"""
    privacy_utility_sweep(make_generator, table, epsilons, metric_fn;
                          n = nothing, hints = ColumnHint[],
                          identifiers = Symbol[],
                          fill = Dict{Symbol, FillSpec}(),
                          rng = Random.default_rng(),
                          delta = 1e-5,
                          metric_kw...) -> Vector{NamedTuple}

Run a privacy–utility curve: for each ε in `epsilons`, build a generator with
`PrivacyBudget(; epsilon = ε, delta)`, fit it to `table`, sample `n` rows
(default: as many as `table` has), and evaluate `metric_fn(real, synth)`.

`make_generator` is anything callable with a `PrivacyBudget`. A private
generator's *type* already is, so the common case reads:

```julia
results = privacy_utility_sweep(
    MSTGenerator, data, [0.1, 0.5, 1.0, 5.0, 10.0], fidelity_score)
```

and a generator needing other settings is a one-line closure:

```julia
privacy_utility_sweep(b -> DiffusionGenerator(privacy = b, epochs = 50),
                      data, epsilons, fidelity_score)
```

A budget is a property of the generator rather than of `fit`, so a sweep over ε
has to build one generator per ε — hence a constructor here rather than a
single instance, which would already have had its ε fixed.

Returns `(; epsilon, delta, metric_result)` per ε, ascending.
"""
function privacy_utility_sweep(make_generator, table, epsilons::AbstractVector,
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

    # A fitted-in generator has already fixed its ε, so sweeping it is
    # meaningless. Say so, rather than failing on a call to a non-callable.
    make_generator isa DataMimic.AbstractGenerator && throw(ArgumentError(
        "pass a generator TYPE or a function of a PrivacyBudget, not a " *
        "constructed generator: its ε is already fixed. Try " *
        "`privacy_utility_sweep($(nameof(typeof(make_generator))), table, " *
        "epsilons, metric_fn)`."))

    cols = Tables.columns(table)
    col_names = collect(Symbol, Tables.columnnames(cols))
    n_rows = length(Tables.getcolumn(cols, first(col_names)))
    sample_n = isnothing(n) ? n_rows : n

    sorted_eps = sort(collect(Float64, epsilons))
    results = NamedTuple[]

    for ε in sorted_eps
        budget = PrivacyBudget(; epsilon = ε, delta = delta)
        generator = make_generator(budget)

        # Sweeping a generator that ignores the budget would produce a flat
        # curve that looks like a finding rather than a mistake.
        DataMimic.privacy_budget(generator) === nothing && throw(ArgumentError(
            "make_generator returned $(nameof(typeof(generator))), which " *
            "spends no privacy budget; a privacy–utility curve over it would " *
            "be constant."))

        model = DataMimic.fit(generator, table;
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
