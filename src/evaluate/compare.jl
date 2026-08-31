# ─── compare ───────────────────────────────────────────────────────────────
#
# Evaluating several generators on your own data, rather than guessing from a
# heuristic which one suits it.
#
# Three properties this needs, each learned by getting it wrong first:
#
#   1. One engine failing must not abort the run.  A diverged diffusion model
#      would otherwise destroy an overnight sweep of every other engine.
#   2. Repeated seeds, because single runs mislead.  MSTGenerator's utility
#      ratio has a seed standard deviation near 0.06 at ε = 0.5, larger than
#      most differences worth acting on.
#   3. No subsampling of the fitting data.  Private engines improve sharply
#      with row count at fixed ε (MST gains ~0.13 utility from 2k to 15k rows,
#      against copula's ~0.02), so fitting on a subsample would systematically
#      penalise exactly the engines a privacy-conscious user came for.

"""
    compare(generators, table; metrics, n, n_seeds, privacy, ...) -> Vector{NamedTuple}

Fit each generator to `table`, sample from it, and score the result with each
metric.  Returns one row per generator × metric, carrying the mean and standard
deviation across seeds and the mean fit time.

The result is a `Vector{NamedTuple}`, which is already a Tables.jl table — pass
it to `DataFrame` to sort or pivot.

# Arguments
- `generators`: the generators to compare.  Entries may also be
  `"label" => generator` to name a row explicitly; otherwise labels are derived
  from the type and, for single-field generators, the field value.

# Keywords
- `metrics`: a `NamedTuple` of `name => f(real, synth)`.  Each function returns
  a number, or a `NamedTuple` from which a scalar is taken (see `metric_field`).
  Defaults to aggregate fidelity and median DCR.
- `metric_field = :aggregate`: which field to read when a metric returns a
  `NamedTuple`.  Falls back to `:ratio`, `:median`, `:mean`, `:score`.
- `n = nothing`: rows to sample from each fitted model (default: as many as
  `table` has).
- `n_seeds = 3`: how many times to repeat each fit and sample.  Fewer than 3
  makes the reported standard deviation meaningless.
- `privacy = nothing`: budget for private generators.  Public generators are
  fitted without one, so a mixed list works in a single call.
- `hints`, `identifiers`, `fill`: forwarded to `fit`.
- `rng`: seeds derive from this, so a run is reproducible.

# Failure handling
A generator that fails on every seed is reported with `ok = false` and its
error message, and the comparison continues with the others.  Partial failures
are counted in `n_failed`, and the seeds that succeeded are still summarised.

# On subsampling
`compare` never fits on a subsample of `table`, however large it is.  Engines
respond differently to row count: a differentially private engine at fixed ε
improves substantially with more rows, because its noise is fixed while the
signal grows.  Fitting on a subsample can therefore reverse the very ranking
the comparison exists to establish.  Subsample deliberately before calling if
you want speed, and read the result as being about that smaller dataset.

# Example
```julia
compare([CopulaGenerator(), CopulaGenerator(:gaussian)], df;
        metrics = (fidelity = fidelity_score,
                   utility  = (r, s) -> utility_tstr(r, s, :income).ratio),
        n_seeds = 5)
```
"""
function compare(generators, table;
                 metrics = (fidelity = fidelity_score,
                            dcr      = privacy_dcr),
                 metric_field::Symbol = :aggregate,
                 n::Union{Nothing, Int} = nothing,
                 n_seeds::Int = 3,
                 privacy::Union{Nothing, PrivacyBudget} = nothing,
                 hints::Vector{ColumnHint} = ColumnHint[],
                 identifiers::Vector{Symbol} = Symbol[],
                 fill = Dict{Symbol, FillSpec}(),
                 rng::AbstractRNG = Random.default_rng())

    Tables.istable(table) ||
        throw(ArgumentError("table must be a Tables.jl-compatible table"))
    isempty(generators) &&
        throw(ArgumentError("no generators to compare"))
    n_seeds >= 1 ||
        throw(ArgumentError("n_seeds must be at least 1, got $n_seeds"))
    isempty(metrics) &&
        throw(ArgumentError("no metrics to evaluate"))

    cols  = Tables.columns(table)
    nrows = length(Tables.getcolumn(cols, first(Tables.columnnames(cols))))
    n_out = something(n, nrows)

    n_seeds < 3 && @warn "n_seeds = $n_seeds; the reported standard deviation " *
                         "will not be meaningful. Engine-to-engine differences " *
                         "can be smaller than seed-to-seed variation."

    base = rand(rng, 1:10^6)
    rows = NamedTuple[]
    labels = _compare_labels(generators)

    for (gen, label) in zip(_compare_gens(generators), labels)
        # Private generators require a budget and public ones reject it, so the
        # budget is passed only where it belongs and a mixed list works.
        gen_privacy = _wants_privacy(gen) ? privacy : nothing

        scores   = Dict{Symbol, Vector{Float64}}(k => Float64[] for k in keys(metrics))
        times    = Float64[]
        n_failed = 0
        first_err = ""

        for seed in 1:n_seeds
            try
                t0 = time()
                model = DataMimic.fit(gen, table;
                                      privacy = gen_privacy, hints = hints,
                                      identifiers = identifiers, fill = fill,
                                      rng = Random.MersenneTwister(base + seed))
                elapsed = time() - t0
                synth = DataMimic.sample(model, n_out;
                            rng = Random.MersenneTwister(base + seed + 10^6))
                push!(times, elapsed)
                for (name, f) in pairs(metrics)
                    push!(scores[name], _metric_value(f(table, synth), metric_field))
                end
            catch e
                n_failed += 1
                isempty(first_err) && (first_err = _short_error(e))
            end
        end

        for name in keys(metrics)
            v = scores[name]
            push!(rows, (;
                generator = label,
                metric    = name,
                ok        = !isempty(v),
                mean      = isempty(v) ? NaN : _cmp_mean(v),
                sd        = length(v) < 2 ? NaN : _cmp_sd(v),
                fit_secs  = isempty(times) ? NaN : _cmp_mean(times),
                n_seeds   = length(v),
                n_failed  = n_failed,
                error     = first_err,
            ))
        end
    end

    return rows
end


# Labels for the results table.
#
# Comparing variants of one engine is the common case, so labelling by type
# name alone would render `CopulaGenerator(:beta)` and
# `CopulaGenerator(:gaussian)` identically and make the output unreadable.
# Single-field generators carry their field; anything still ambiguous is
# numbered; and `"label" => generator` pairs override entirely.

_compare_gens(generators) = [g isa Pair ? last(g) : g for g in generators]

"""Type name, carrying the field value when a generator has exactly one."""
function _auto_label(gen)
    T = typeof(gen)
    name = string(nameof(T))
    fields = fieldnames(T)
    length(fields) == 1 || return name
    v = getfield(gen, first(fields))
    return string(name, "(", v isa Symbol ? ":" * string(v) : string(v), ")")
end

function _compare_labels(generators)
    labels = [g isa Pair ? string(first(g)) : _auto_label(g) for g in generators]
    # Number any that are still identical, so every row is addressable.
    counts = Dict{String, Int}()
    for l in labels
        counts[l] = get(counts, l, 0) + 1
    end
    seen = Dict{String, Int}()
    return map(labels) do l
        counts[l] == 1 && return l
        seen[l] = get(seen, l, 0) + 1
        string(l, " #", seen[l])
    end
end

# Which generators need the budget handed to them.
_wants_privacy(::DataMimic.AbstractPrivateGenerator) = true
_wants_privacy(gen::DataMimic.DiffusionGenerator)    = gen.dp
_wants_privacy(::Any)                                = false

_cmp_mean(v) = sum(v) / length(v)
_cmp_sd(v)   = sqrt(sum(abs2, v .- _cmp_mean(v)) / (length(v) - 1))

"""Pull a scalar out of whatever a metric returned."""
function _metric_value(result, field::Symbol)
    result isa Number && return Float64(result)
    if result isa NamedTuple
        hasproperty(result, field) && return Float64(getproperty(result, field))
        for f in (:aggregate, :ratio, :median, :mean, :score)
            hasproperty(result, f) && return Float64(getproperty(result, f))
        end
        throw(ArgumentError(
            "metric returned a NamedTuple with fields $(keys(result)); none " *
            "is a recognised scalar. Pass metric_field, or wrap the metric so " *
            "it returns a number."))
    end
    throw(ArgumentError(
        "metric must return a number or a NamedTuple, got $(typeof(result))"))
end

"""First line of an error message, truncated, for the results table."""
function _short_error(e)
    line = first(split(sprint(showerror, e), '\n'))
    return length(line) > 160 ? line[1:157] * "..." : line
end
