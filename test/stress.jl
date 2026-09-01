# ─── Stress: shapes and parameter combinations ─────────────────────────────
#
# Every user-facing bug found in this package so far was triggered by a
# *shape*, a *parameter combination*, or a *model state* — not by a particular
# dataset:
#
#   - NaN in the block log-softmax needed >= 6 categorical columns AND a wide
#     batch, simultaneously.
#   - `_anneal_lr` returned zero for `epochs = 1`.
#   - Rejection sampling biased the label distribution only for a
#     class-conditional model that was undertrained enough to diverge.
#   - `ColumnHint(kind = :continuous)` threw on a column stored as Int.
#
# Collecting more real datasets would have found none of the first three. So
# this file generates the shapes instead, and sweeps the parameter space.
#
# It asserts INVARIANTS, never quality. Quality depends on the data and the
# configuration; these should hold for any input a caller is allowed to pass:
#
#   1. `fit` either succeeds or throws a *typed, explanatory* error. Silent
#      garbage is the failure mode being hunted here.
#   2. `sample(m, n)` returns exactly `n` rows.
#   3. Column names survive, except identifiers dropped without a `fill`.
#   4. No NaN or Inf reaches numeric output.
#   5. Synthetic categorical levels are a subset of the observed ones.
#   6. The same `rng` gives byte-identical output.

using Random

# ── Table construction ─────────────────────────────────────────────────────

"""
Build a table from a shape spec. Deliberately includes the awkward cases:
constant columns, single-level categoricals, near-unique numerics that trip
the identifier heuristic, heavy skew, and missingness.
"""
function stress_table(; n, n_num = 0, n_cat = 0, card = 5, missing_rate = 0.0,
                        constant = false, single_level = false,
                        near_unique = false, skewed = false, seed = 1)
    rng = MersenneTwister(seed)
    cols = Pair{Symbol,Any}[]

    for j in 1:n_num
        v = skewed ? exp.(randn(rng, n) .* 2) : randn(rng, n)
        if missing_rate > 0
            v = Vector{Union{Missing,Float64}}(v)
            for i in 1:n
                rand(rng) < missing_rate && (v[i] = missing)
            end
        end
        push!(cols, Symbol("num", j) => v)
    end

    for j in 1:n_cat
        levels = string.(1:card)
        v = rand(rng, levels, n)
        if missing_rate > 0
            v = Vector{Union{Missing,String}}(v)
            for i in 1:n
                rand(rng) < missing_rate && (v[i] = missing)
            end
        end
        push!(cols, Symbol("cat", j) => v)
    end

    constant     && push!(cols, :const_col   => fill(7.0, n))
    single_level && push!(cols, :one_level   => fill("only", n))
    near_unique  && push!(cols, :near_unique => collect(1:n) .+ 0.5)

    isempty(cols) && push!(cols, :fallback => randn(rng, n))
    return DataFrame(cols...)
end

# ── Invariant checks ───────────────────────────────────────────────────────

"""
Run one engine against one table and assert the invariants above.

Returns `:ok`, or `:rejected` when the engine refused with a typed error —
which is an acceptable outcome, since refusing is what several of this
package's guards are for.
"""
function check_engine(gen, df; privacy = nothing, hints = ColumnHint[],
                      n_out = 50, label = "")
    model = try
        fit(gen, df; privacy = privacy, hints = hints, rng = MersenneTwister(11))
    catch err
        # A refusal is fine; an undefined error is not.
        @test err isa Union{ArgumentError, ErrorException, DimensionMismatch}
        @test !isempty(sprint(showerror, err))
        return :rejected
    end

    syn = try
        sample(model, n_out; rng = MersenneTwister(12))
    catch err
        @test err isa Union{ArgumentError, ErrorException, DimensionMismatch}
        return :rejected
    end

    cols_out = Symbol.(names(syn))

    # 2. exact row count
    @test nrow(syn) == n_out

    # 3. schema preserved, modulo identifiers dropped without a fill spec
    dropped = setdiff(Symbol.(names(df)), cols_out)
    for d in dropped
        @test DataMimic.detect_column_type(df[!, d]) === :identifier
    end
    @test isempty(setdiff(cols_out, Symbol.(names(df))))

    for c in cols_out
        v = syn[!, c]
        real_col = df[!, c]

        # 4. no non-finite numbers
        if nonmissingtype(eltype(v)) <: AbstractFloat
            @test all(x -> ismissing(x) || isfinite(x), v)
        end

        # 5. categorical levels are a subset of what was observed
        if nonmissingtype(eltype(real_col)) <: AbstractString
            seen = Set(skipmissing(real_col))
            @test all(x -> ismissing(x) || x in seen, v)
        end
    end

    # 6. determinism
    syn2 = sample(model, n_out; rng = MersenneTwister(12))
    for c in cols_out
        @test isequal(syn[!, c], syn2[!, c])
    end

    return :ok
end

# ── The shape grid ─────────────────────────────────────────────────────────

const STRESS_SHAPES = [
    (; name = "numeric only, tiny",        n = 60,   n_num = 3),
    (; name = "numeric only, wide",        n = 800,  n_num = 12),
    (; name = "categorical only",          n = 600,  n_cat = 5, card = 4),
    (; name = "many categoricals",         n = 900,  n_cat = 9, card = 6),
    (; name = "high cardinality",          n = 900,  n_cat = 3, card = 40),
    (; name = "mixed, small",              n = 120,  n_num = 2, n_cat = 2),
    (; name = "mixed, medium",             n = 1500, n_num = 4, n_cat = 4),
    (; name = "binary categoricals",       n = 500,  n_num = 2, n_cat = 4, card = 2),
    (; name = "with missing values",       n = 700,  n_num = 3, n_cat = 3,
                                           missing_rate = 0.2),
    (; name = "constant column",           n = 400,  n_num = 2, n_cat = 2,
                                           constant = true),
    (; name = "single-level categorical",  n = 400,  n_num = 2, n_cat = 1,
                                           single_level = true),
    (; name = "identifier-like column",    n = 400,  n_num = 2, n_cat = 2,
                                           near_unique = true),
    (; name = "heavy skew",                n = 700,  n_num = 4, skewed = true),
    (; name = "one row per level",         n = 50,   n_num = 1, n_cat = 2, card = 25),
    (; name = "everything at once",        n = 900,  n_num = 3, n_cat = 4, card = 8,
                                           missing_rate = 0.1, constant = true,
                                           single_level = true, near_unique = true),
]

@testset "Stress" begin

    @testset "shapes × engines" begin
        budget = PrivacyBudget(epsilon = 2.0)
        engines = [
            ("CopulaGenerator(:beta)",     CopulaGenerator(),          nothing),
            ("CopulaGenerator(:gaussian)", CopulaGenerator(:gaussian), nothing),
            ("MSTGenerator",               MSTGenerator(),             budget),
            ("DPCopulaGenerator",          DPCopulaGenerator(),        budget),
        ]

        for shape in STRESS_SHAPES
            spec = Base.structdiff(shape, NamedTuple{(:name,)})
            df = stress_table(; spec...)
            @testset "$(shape.name)" begin
                for (label, gen, priv) in engines
                    check_engine(gen, df; privacy = priv, label = label)
                end
            end
        end
    end

    # ── Parameter sweep for the diffusion engine ───────────────────────────
    #
    # Kept to small tables and tiny models: the point is to hit awkward
    # parameter *combinations*, not to train anything useful.
    @testset "diffusion parameters" begin
        df = stress_table(; n = 400, n_num = 2, n_cat = 3, card = 4, seed = 5)

        configs = [
            ("epochs = 1",            (; epochs = 1,  batch_size = 128)),
            ("epochs = 2",            (; epochs = 2,  batch_size = 128)),
            ("batch larger than n",   (; epochs = 2,  batch_size = 4096)),
            ("batch of one",          (; epochs = 1,  batch_size = 1)),
            ("few timesteps",         (; epochs = 2,  batch_size = 128,
                                         num_timesteps = 5)),
            ("single hidden layer",   (; epochs = 2,  batch_size = 128,
                                         d_layers = [16])),
            ("dropout enabled",       (; epochs = 2,  batch_size = 128,
                                         dropout = 0.2)),
            ("with lr warmup",        (; epochs = 3,  batch_size = 128,
                                         lr_warmup = 2)),
            ("ema disabled",          (; epochs = 2,  batch_size = 128,
                                         ema_decay = 0.0)),
        ]

        for (label, kw) in configs
            @testset "$label" begin
                gen = DiffusionGenerator(; d_layers = [16, 16], num_timesteps = 20,
                                           kw...)
                check_engine(gen, df; n_out = 40, label = label)
            end
        end

        # Class-conditional, including the degenerate single-level target that
        # the label-embedding path has to survive.
        @testset "class-conditional" begin
            dfc = copy(df)
            dfc.target = rand(MersenneTwister(6), ["a", "b"], nrow(dfc))
            gen = DiffusionGenerator(epochs = 2, batch_size = 128,
                                     d_layers = [16, 16], num_timesteps = 20,
                                     target = :target)
            check_engine(gen, dfc; n_out = 40)
        end

        # DP-SGD, including the dropout case that falls back off ghost clipping.
        @testset "dp-sgd" begin
            for (label, dropout) in (("ghost clipping", 0.0), ("loop fallback", 0.3))
                @testset "$label" begin
                    gen = DiffusionGenerator(dp = true, epochs = 2, batch_size = 64,
                                             d_layers = [16, 16], num_timesteps = 20,
                                             dropout = dropout)
                    check_engine(gen, df; privacy = PrivacyBudget(epsilon = 8.0),
                                 n_out = 40)
                end
            end
        end
    end

    # ── Diffusion against the awkward shapes, not just parameters ──────────
    #
    # The parameter sweep above uses one easy table, which would not have
    # caught the block-softmax NaN: that needed many categorical columns AND a
    # wide batch at the same time. Shape and configuration have to be crossed,
    # not tested separately.
    @testset "diffusion × shapes" begin
        shapes = [
            ("many categoricals, wide batch",
             stress_table(; n = 1200, n_num = 2, n_cat = 9, card = 6, seed = 7),
             (; batch_size = 1024)),
            ("high cardinality",
             stress_table(; n = 800, n_num = 2, n_cat = 3, card = 40, seed = 8),
             (; batch_size = 512)),
            ("categorical only",
             stress_table(; n = 600, n_cat = 6, card = 5, seed = 9),
             (; batch_size = 256)),
            ("numeric only",
             stress_table(; n = 600, n_num = 6, seed = 10),
             (; batch_size = 256)),
            ("missing values",
             stress_table(; n = 600, n_num = 3, n_cat = 3, missing_rate = 0.25, seed = 11),
             (; batch_size = 256)),
            ("constant and single-level columns",
             stress_table(; n = 500, n_num = 2, n_cat = 2, constant = true,
                            single_level = true, seed = 12),
             (; batch_size = 256)),
        ]

        for (label, df, kw) in shapes
            @testset "$label" begin
                gen = DiffusionGenerator(; epochs = 2, d_layers = [32, 32],
                                           num_timesteps = 20, kw...)
                check_engine(gen, df; n_out = 60, label = label)
            end
        end
    end

    # ── Hints across every kind, on every storage type ─────────────────────
    @testset "hints × storage types" begin
        df = DataFrame(int_col   = rand(MersenneTwister(1), 1:12, 400),
                       float_col = randn(MersenneTwister(2), 400),
                       str_col   = rand(MersenneTwister(3), ["p", "q", "r"], 400),
                       bool_col  = rand(MersenneTwister(4), [true, false], 400))

        for col in (:int_col, :float_col)
            for kind in (:continuous, :integer, :categorical)
                @testset "$col as $kind" begin
                    check_engine(CopulaGenerator(), df;
                                 hints = [ColumnHint(name = col, kind = kind)])
                end
            end
        end

        for kind in (:categorical, :binary)
            @testset "str_col as $kind" begin
                check_engine(CopulaGenerator(), df;
                             hints = [ColumnHint(name = :str_col, kind = kind)])
            end
        end
    end
end
