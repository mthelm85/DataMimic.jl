using DataMimic
using DataFrames
using Test
using Random
using LinearAlgebra: eigvals
using Statistics: cor
using Lux, Zygote

# A generator that always fails, used to check that compare() isolates a
# broken engine instead of letting it abort the whole comparison.
struct BoomGenerator <: DataMimic.AbstractGenerator end
DataMimic._fit_engine(::BoomGenerator, args...) = error("engine exploded")
DataMimic.privacy_budget(::BoomGenerator) = nothing

@testset "DataMimic.jl" begin

    # ── Helpers ──────────────────────────────────────────────────────────────
    function make_df(n = 200)
        DataFrame(
            x_float = randn(n),
            x_int   = rand(1:50, n),
            x_cat   = rand(["a", "b", "c"], n),
            x_bool  = rand([true, false], n),
            x_const = fill("same", n),
        )
    end

    # ════════════════════════════════════════════════════════════════════════
    # Type System
    # ════════════════════════════════════════════════════════════════════════
    @testset "Type System" begin
        @testset "PrivacyBudget" begin
            pb = PrivacyBudget(epsilon = 1.0)
            @test pb.epsilon == 1.0
            @test pb.delta == 1e-5

            pb2 = PrivacyBudget(epsilon = 0.5, delta = 1e-3)
            @test pb2.delta == 1e-3

            @test_throws ArgumentError PrivacyBudget(epsilon = -1.0)
            @test_throws ArgumentError PrivacyBudget(epsilon = 0.0)
            @test_throws ArgumentError PrivacyBudget(epsilon = 1.0, delta = 1.0)
            @test_throws ArgumentError PrivacyBudget(epsilon = 1.0, delta = -0.1)
        end

        @testset "CopulaGenerator" begin
            @test CopulaGenerator().copula_type == :beta
            @test CopulaGenerator(:gaussian).copula_type == :gaussian
            @test_throws ArgumentError CopulaGenerator(:invalid)
        end

        @testset "MSTGenerator" begin
            # Marginal order is not a parameter: MST is the 2-way spanning
            # tree. The budget is the only thing to configure, and it is
            # required - a private generator without one cannot run, so the
            # type refuses to represent that state at all.
            g = MSTGenerator(ε = 1.0)
            @test g isa DataMimic.AbstractPrivateGenerator
            @test fieldnames(MSTGenerator) == (:privacy,)
            @test privacy_budget(g).epsilon == 1.0
            @test MSTGenerator(PrivacyBudget(ε = 1.0)) == g
            @test_throws ArgumentError MSTGenerator()        # no budget
            @test_throws MethodError   MSTGenerator(2)       # not a budget
            # Marginal order is gone for good.
            @test_throws MethodError MSTGenerator(2, 3)
        end

        @testset "DiffusionGenerator" begin
            dg = DiffusionGenerator()
            @test dg.epochs == 100
            @test dg.batch_size == 512
            @test dg.privacy === nothing   # no budget = no DP-SGD
            @test_throws ArgumentError DiffusionGenerator(epochs = 0)
            @test_throws ArgumentError DiffusionGenerator(batch_size = 0)
        end

        @testset "ColumnHint" begin
            h = ColumnHint(name = :x, kind = :continuous)
            @test h.name == :x
            @test h.kind == :continuous
            @test h.levels === nothing

            h2 = ColumnHint(name = :y, kind = :categorical, levels = ["a", "b"])
            @test h2.levels == ["a", "b"]

            @test_throws ArgumentError ColumnHint(name = :x, kind = :invalid)
        end

        @testset "Phase 2 model types" begin
            @test FittedMSTModel <: AbstractFittedModel
            @test FittedDPCopulaModel <: AbstractFittedModel
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Column Detection
    # ════════════════════════════════════════════════════════════════════════
    @testset "detect_column_type" begin
        using DataMimic: detect_column_type

        # ── Constant ─────────────────────────────────────────────────────
        @test detect_column_type(fill(1.0, 5))         == :constant
        @test detect_column_type(fill("x", 10))        == :constant
        @test detect_column_type(fill(42, 3))           == :constant

        # Entirely missing → constant
        @test detect_column_type(Union{Float64,Missing}[missing, missing]) == :constant

        # ── Bool is always binary ────────────────────────────────────────
        @test detect_column_type([true, false, true])   == :binary
        @test detect_column_type(Bool[true, true, true]) == :binary  # 1 unique but Bool type

        # ── String / Symbol ──────────────────────────────────────────────
        @test detect_column_type(["a", "b"])            == :binary
        @test detect_column_type(["a", "b", "c"])       == :categorical
        @test detect_column_type([:x, :y, :z])          == :categorical

        # ── Float: continuous vs whole-number ────────────────────────────
        @test detect_column_type([1.1, 2.2, 3.3])       == :continuous
        @test detect_column_type([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]) == :continuous
        @test detect_column_type([1.0, 2.0])             == :binary   # 2 whole-number floats
        @test detect_column_type([1.0, 2.0, 3.0])        == :integer  # 3 unique, n=3, threshold=2
        @test detect_column_type([1.0, 2.0, 3.0, 4.0])  == :integer

        # NaN / Inf are filtered out, remaining values classified
        @test detect_column_type([1.1, NaN, 3.3, 4.4])  == :continuous
        @test detect_column_type([NaN, Inf, -Inf])       == :constant  # nothing left

        # ── Integer: cardinality-aware ───────────────────────────────────
        # Small samples — threshold = max(2, n÷20), capped at 20
        @test detect_column_type([1, 2, 3])              == :integer  # n=3, thresh=2, 3>2
        @test detect_column_type(Int[1, 2, 3, 4])        == :integer  # n=4, thresh=2, 4>2

        # Low-cardinality integers in larger samples → categorical
        # 1000 rows, 3 unique → threshold = min(20, max(2, 50)) = 20, 3 ≤ 20 → categorical
        @test detect_column_type(rand(1:3, 1000))        == :categorical

        # High-cardinality integers → integer
        # 1000 rows, ~50 unique → 50 > 20 → integer
        @test detect_column_type(rand(1:100, 1000))      == :integer

        # ── Missing values mixed in ──────────────────────────────────────
        col_miss = Union{Float64,Missing}[1.0, missing, 3.0]
        @test detect_column_type(col_miss) == :binary  # 2 non-missing unique whole floats

        col_cont = Union{Float64,Missing}[1.1, missing, 3.3, 4.4]
        @test detect_column_type(col_cont) == :continuous

        col_int = Union{Int,Missing}[1, missing, 2, 3, 4, 5]
        @test detect_column_type(col_int) == :integer  # n_nm=5, thresh=2, 5>2
    end

    # ════════════════════════════════════════════════════════════════════════
    # fit / CopulaGenerator
    # ════════════════════════════════════════════════════════════════════════
    @testset "fit with CopulaGenerator" begin
        df = make_df()
        model = fit(CopulaGenerator(), df)

        @test model isa FittedCopulaModel
        @test :x_float in model.column_names
        @test model.n_original == 200
        @test !isnothing(model.copula)
        @test :x_float in model.copula_columns
        @test :x_int   in model.copula_columns

        # Column kinds
        idx_float = findfirst(==(:x_float), model.column_names)
        idx_cat   = findfirst(==(:x_cat),   model.column_names)
        idx_const = findfirst(==(:x_const), model.column_names)
        idx_bool  = findfirst(==(:x_bool),  model.column_names)
        @test model.column_kinds[idx_float] == :continuous
        @test model.column_kinds[idx_cat]   == :categorical
        @test model.column_kinds[idx_const] == :constant
        @test model.column_kinds[idx_bool]  == :binary

        # All missingness should be 0
        @test all(v -> v == 0.0, values(model.missingness))
    end

    # ════════════════════════════════════════════════════════════════════════
    # sample
    # ════════════════════════════════════════════════════════════════════════
    @testset "sample output" begin
        df    = make_df(100)
        model = fit(CopulaGenerator(), df)
        syn   = sample(model, 150)

        @test syn isa DataFrame
        @test nrow(syn) == 150
        @test ncol(syn) == 5
        @test names(syn) == names(df)

        # Float column stays Float64
        @test eltype(syn.x_float) == Float64

        # Integer column has no fractional values
        @test all(v -> v == round(v), syn.x_int)

        # Categorical column only contains original levels
        original_levels = Set(df.x_cat)
        @test all(v -> v in original_levels, syn.x_cat)

        # Boolean output only contains true/false
        @test all(v -> v isa Bool, syn.x_bool)

        # Constant column preserved
        @test all(==("same"), syn.x_const)
    end

    # ════════════════════════════════════════════════════════════════════════
    # Low-cardinality integers as categorical
    # ════════════════════════════════════════════════════════════════════════
    @testset "low-cardinality integer → categorical" begin
        n = 500
        df = DataFrame(
            encoded = rand(1:3, n),     # 3 levels → categorical
            real_num = randn(n),
        )
        model = fit(CopulaGenerator(), df)
        idx = findfirst(==(:encoded), model.column_names)
        @test model.column_kinds[idx] == :categorical
        # REQ-CPL-006: categoricals join the copula via ordinal encoding
        @test :encoded in model.copula_columns

        syn = sample(model, 200)
        @test all(v -> v in [1, 2, 3], syn.encoded)
    end

    # ════════════════════════════════════════════════════════════════════════
    # Identifiers
    # ════════════════════════════════════════════════════════════════════════
    @testset "identifiers" begin
        n  = 200
        df = DataFrame(
            id     = ["user_$i" for i in 1:n],
            amount = randn(n),
            cat    = rand(["a", "b", "c"], n),
        )

        @testset "drop by default" begin
            model = fit(CopulaGenerator(), df; identifiers = [:id])
            syn   = sample(model, 100)
            @test !(:id in Symbol.(names(syn)))
            @test :amount in Symbol.(names(syn))
            @test nrow(syn) == 100
        end

        @testset "sequential fill" begin
            model = fit(CopulaGenerator(), df;
                identifiers = [:id],
                fill = Dict(:id => :sequential))
            syn = sample(model, 100)
            @test :id in Symbol.(names(syn))
            @test syn.id[1] == "id_1"
            @test syn.id[100] == "id_100"
        end

        @testset "sequential_int fill" begin
            model = fit(CopulaGenerator(), df;
                identifiers = [:id],
                fill = Dict(:id => :sequential_int))
            syn = sample(model, 50)
            @test syn.id == collect(1:50)
        end

        @testset "string prefix fill" begin
            model = fit(CopulaGenerator(), df;
                identifiers = [:id],
                fill = Dict(:id => "person"))
            syn = sample(model, 3)
            @test syn.id == ["person_1", "person_2", "person_3"]
        end

        @testset "function fill" begin
            model = fit(CopulaGenerator(), df;
                identifiers = [:id],
                fill = Dict(:id => i -> "SYNTH-$(lpad(i, 5, '0'))"))
            syn = sample(model, 2)
            @test syn.id == ["SYNTH-00001", "SYNTH-00002"]
        end

        @testset "auto-detection" begin
            model = fit(CopulaGenerator(), df)
            @test :id in model.identifier_columns
            syn = sample(model, 50)
            @test !(:id in Symbol.(names(syn)))
        end

        @testset "hint overrides auto-detection" begin
            model = fit(CopulaGenerator(), df;
                hints = [ColumnHint(name = :id, kind = :categorical)])
            @test !(:id in model.identifier_columns)
        end

        @testset "all identifiers → error" begin
            small = DataFrame(id = ["a", "b", "c"])
            @test_throws ArgumentError fit(CopulaGenerator(), small;
                identifiers = [:id])
        end

        @testset "fill key not an identifier → error" begin
            @test_throws ArgumentError fit(CopulaGenerator(), df;
                fill = Dict(:amount => :sequential))
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Tables.jl round-trip (Phase 1 — Copula)
    # ════════════════════════════════════════════════════════════════════════
    @testset "Tables.jl round-trip" begin
        @testset "NamedTuple in → NamedTuple out" begin
            nt = (x = randn(100), y = rand(1:50, 100),
                  z = rand(["a", "b"], 100))
            model = fit(CopulaGenerator(), nt)
            syn   = sample(model, 50)
            @test syn isa NamedTuple
            @test length(syn.x) == 50
            @test length(syn.y) == 50
            @test length(syn.z) == 50
        end

        @testset "DataFrame in → DataFrame out" begin
            df    = DataFrame(x = randn(100), y = rand(1:50, 100))
            model = fit(CopulaGenerator(), df)
            syn   = sample(model, 50)
            @test syn isa DataFrame
            @test nrow(syn) == 50
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Privacy validation
    # ════════════════════════════════════════════════════════════════════════
    @testset "PrivacyBudget accepts ε/δ" begin
        # Greek spellings are aliases, not a second parameterization.
        @test PrivacyBudget(ε = 1.0, δ = 1e-5) == PrivacyBudget(epsilon = 1.0, delta = 1e-5)
        @test PrivacyBudget(ε = 1.0).delta == 1e-5          # same δ default
        @test PrivacyBudget(ε = 2.0).epsilon == 2.0

        # Either spelling reads back, whichever was used to build it.
        b = PrivacyBudget(ε = 1.5, δ = 1e-6)
        @test b.epsilon == b.ε == 1.5
        @test b.delta == b.δ == 1e-6
        @test Set(propertynames(b)) == Set((:epsilon, :delta, :ε, :δ))

        # Supplying both spellings is a mistake, not a precedence puzzle.
        @test_throws ArgumentError PrivacyBudget(epsilon = 1.0, ε = 2.0)
        @test_throws ArgumentError PrivacyBudget(ε = 1.0, delta = 1e-5, δ = 1e-6)
        @test_throws ArgumentError PrivacyBudget(δ = 1e-5)   # no epsilon at all

        # Validation still applies through the alias path.
        @test_throws ArgumentError PrivacyBudget(ε = -1.0)
        @test_throws ArgumentError PrivacyBudget(ε = 1.0, δ = 2.0)

        # And a budget built either way still drives a private fit.
        df = DataFrame(a = randn(MersenneTwister(1), 300),
                       b = rand(MersenneTwister(2), ["x", "y", "z"], 300))
        m = fit(MSTGenerator(ε = 1.0), df; rng = MersenneTwister(3))
        @test nrow(sample(m, 50; rng = MersenneTwister(4))) == 50
    end

    @testset "privacy is a property of the generator" begin
        df = make_df()
        pb = PrivacyBudget(epsilon = 1.0)

        # These used to be three `_validate_privacy` methods checking at fit
        # time what construction can now make impossible.

        # A public generator has nowhere to put a budget, so the mistake is a
        # dispatch failure rather than a runtime check inside fit.
        @test privacy_budget(CopulaGenerator()) === nothing
        @test_throws MethodError CopulaGenerator(privacy = pb)

        # A private generator cannot be built without one.
        @test_throws ArgumentError MSTGenerator()
        @test_throws ArgumentError DPCopulaGenerator()

        # And fit no longer accepts a budget at all.
        @test_throws MethodError fit(MSTGenerator(privacy = pb), df;
                                     privacy = pb)

        # The budget the generator carries is the one that gets spent.
        @test privacy_budget(MSTGenerator(privacy = pb)) === pb
        @test privacy_budget(DPCopulaGenerator(ε = 2.0)).epsilon == 2.0
    end

    # ════════════════════════════════════════════════════════════════════════
    # Missingness
    # ════════════════════════════════════════════════════════════════════════
    @testset "missingness" begin
        n   = 500
        col = Vector{Union{Float64, Missing}}(randn(n))
        col[1:50] .= missing
        df    = DataFrame(x = col, y = randn(n))
        model = fit(CopulaGenerator(), df)

        @test isapprox(model.missingness[:x], 0.1, atol = 0.01)

        syn = sample(model, 1000)
        p_obs = count(ismissing, syn.x) / 1000
        @test isapprox(p_obs, 0.1, atol = 0.05)
        @test Missing <: eltype(syn.x)
    end

    # ════════════════════════════════════════════════════════════════════════
    # NaN / Inf handling
    # ════════════════════════════════════════════════════════════════════════
    @testset "NaN and Inf in float columns" begin
        n  = 100
        xs = randn(n)
        xs[1]  = NaN
        xs[2]  = Inf
        xs[3]  = -Inf
        df = DataFrame(x = xs, y = randn(n))
        model = fit(CopulaGenerator(), df)

        # The 3 non-finite values should be treated as missing
        @test model.missingness[:x] > 0.0

        syn = sample(model, 50)
        @test nrow(syn) == 50
        # Sampled values should all be finite (or missing)
        @test all(v -> ismissing(v) || isfinite(v), syn.x)
    end

    # ════════════════════════════════════════════════════════════════════════
    # Serialization (Phase 1 — Copula)
    # ════════════════════════════════════════════════════════════════════════
    @testset "save / load (Copula)" begin
        df    = make_df()
        model = fit(CopulaGenerator(), df)

        path = tempname() * ".dmimic"
        save(path, model)
        loaded = load(path)

        @test loaded isa FittedCopulaModel
        @test loaded.n_original == model.n_original
        @test loaded.column_names == model.column_names
        @test loaded.column_kinds == model.column_kinds

        syn = sample(loaded, 50)
        @test syn isa DataFrame
        @test nrow(syn) == 50

        rm(path)
    end

    # ════════════════════════════════════════════════════════════════════════
    # Error conditions
    # ════════════════════════════════════════════════════════════════════════
    @testset "error conditions" begin
        @test_throws ArgumentError fit(CopulaGenerator(), DataFrame(a = Int[]))
        @test_throws ArgumentError fit(CopulaGenerator(), DataFrame())
        @test_throws ArgumentError fit(CopulaGenerator(), "not a table")

        model = fit(CopulaGenerator(), make_df())
        @test_throws ArgumentError sample(model, 0)
        @test_throws ArgumentError sample(model, -1)

        @test_throws ArgumentError fit(CopulaGenerator(), make_df();
            hints = [ColumnHint(name = :bogus, kind = :continuous)])
        @test_throws ArgumentError fit(CopulaGenerator(), make_df();
            identifiers = [:bogus])
    end

    # ════════════════════════════════════════════════════════════════════════
    # synthesize convenience
    # ════════════════════════════════════════════════════════════════════════
    @testset "synthesize" begin
        df  = make_df(50)
        syn = synthesize(CopulaGenerator(), df, 60)
        @test syn isa DataFrame
        @test nrow(syn) == 60
        @test ncol(syn) == 5
        @test names(syn) == names(df)
    end

    # ════════════════════════════════════════════════════════════════════════
    # Single numeric / all categorical fallbacks
    # ════════════════════════════════════════════════════════════════════════
    # REQ-CPL-004: the fallback threshold counts *modellable* columns, not
    # numeric ones — a numeric column plus a categorical is now enough to fit
    # a copula, and an all-categorical table is too.
    @testset "one numeric + one categorical still fits a copula" begin
        df = DataFrame(x = randn(100), cat = rand(["a", "b", "c"], 100))
        model = fit(CopulaGenerator(), df)
        @test !isnothing(model.copula)
        @test Set(model.copula_columns) == Set([:x, :cat])
        syn = sample(model, 50)
        @test nrow(syn) == 50
        @test all(v -> v in ["a", "b", "c"], syn.cat)
    end

    @testset "all categorical fits a copula" begin
        df = DataFrame(a = rand(["x", "y", "z"], 100),
                        b = rand(["p", "q"], 100))
        model = fit(CopulaGenerator(), df)
        @test !isnothing(model.copula)
        syn = sample(model, 30)
        @test nrow(syn) == 30
        @test all(v -> v in ["x", "y", "z"], syn.a)
        @test all(v -> v in ["p", "q"], syn.b)
    end

    @testset "single modellable column falls back with a warning" begin
        df = DataFrame(x = randn(100))
        model = @test_logs (:warn,) fit(CopulaGenerator(), df)
        @test isnothing(model.copula)
        syn = sample(model, 50)
        @test nrow(syn) == 50
    end

    # A categorical that collapsed to one level cannot be encoded, so it is
    # kept out of the copula and drawn independently.
    @testset "single-level categorical is excluded from the copula" begin
        df = DataFrame(x = randn(100), y = randn(100),
                       flat = fill("only", 100))
        model = fit(CopulaGenerator(), df)
        @test !isnothing(model.copula)
        @test !(:flat in model.copula_columns)
        syn = sample(model, 40)
        @test all(==("only"), syn.flat)
    end

    @testset "singular correlation is repaired, not thrown" begin
        # The Gaussian copula factorizes the correlation of the normal scores.
        # Collinear columns, or fewer complete cases than columns, make that
        # matrix singular, and the Cholesky used to escape as a bare
        # PosDefException from inside Distributions. Found by a sweep over
        # OpenML tables: four datasets of 13-15 rows crashed on :gaussian
        # while :beta handled every one of them.
        rng = MersenneTwister(2)

        # Fewer complete cases than columns.
        narrow = DataFrame()
        for j in 1:8
            narrow[!, Symbol("x", j)] = randn(rng, 6)
        end
        m = @test_logs (:warn, r"singular"i) fit(CopulaGenerator(:gaussian), narrow;
                                                 rng = MersenneTwister(1))
        syn = sample(m, 20; rng = MersenneTwister(3))
        @test nrow(syn) == 20
        @test all(isfinite, Matrix(syn))

        # A duplicated column: plenty of rows, still exactly collinear.
        dup = DataFrame()
        for j in 1:4
            dup[!, Symbol("y", j)] = randn(rng, 400)
        end
        dup[!, :y5] = dup.y1

        # Whether an exactly collinear column trips the Cholesky is a matter of
        # round-off - at 400 rows it sometimes factorizes anyway - so assert the
        # outcome here rather than which of the two paths produced it. The
        # narrow table above is the deterministic trigger.
        m2 = fit(CopulaGenerator(:gaussian), dup; rng = MersenneTwister(1))
        s2 = sample(m2, 2000; rng = MersenneTwister(3))
        @test all(isfinite, Matrix(s2))

        # The repair must preserve the dependence, not merely avoid the throw.
        @test cor(s2.y1, s2.y5) > 0.999

        @test isequal(s2, sample(m2, 2000; rng = MersenneTwister(3)))

        # :beta needs no adjustment on either table.
        @test !isnothing(fit(CopulaGenerator(), dup; rng = MersenneTwister(1)).copula)

        # An ordinary table still takes the library path, with no warning.
        plain = DataFrame(a = randn(rng, 200), b = randn(rng, 200),
                          c = rand(rng, ["x", "y", "z"], 200))
        m3 = fit(CopulaGenerator(:gaussian), plain; rng = MersenneTwister(1))
        @test nrow(sample(m3, 50; rng = MersenneTwister(3))) == 50
    end

    # ════════════════════════════════════════════════════════════════════════
    # Reproducibility (RNG)
    # ════════════════════════════════════════════════════════════════════════
    @testset "reproducibility" begin
        df = make_df(100)
        model = fit(CopulaGenerator(), df)

        syn1 = sample(model, 50; rng = MersenneTwister(42))
        syn2 = sample(model, 50; rng = MersenneTwister(42))

        @test syn1.x_float == syn2.x_float
        @test syn1.x_int   == syn2.x_int
        @test syn1.x_cat   == syn2.x_cat
    end

    # ════════════════════════════════════════════════════════════════════════
    #  P H A S E   2  —  P R I V A C Y
    # ════════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════════
    # DP Utilities
    # ════════════════════════════════════════════════════════════════════════
    @testset "DP utilities" begin
        using DataMimic: _eps_delta_to_rho, _rho_to_sigma,
                         _project_psd, _project_correlation

        @testset "zCDP conversion" begin
            rho = _eps_delta_to_rho(1.0, 1e-5)
            @test rho > 0
            @test rho < 1.0     # ρ ≤ ε for reasonable δ
        end

        @testset "sigma from rho" begin
            sigma = _rho_to_sigma(0.5, 1.0)
            @test sigma == 1.0 / sqrt(1.0)   # Δ/√(2ρ) = 1/1
        end

        @testset "PSD projection" begin
            # A matrix with a negative eigenvalue
            M = [1.0 2.0; 2.0 1.0]   # eigenvalues: -1, 3
            P = _project_psd(M)
            @test all(eigvals(P) .> 0)
        end

        @testset "correlation projection" begin
            M = [1.0 0.5; 0.5 1.0]
            C = _project_correlation(M)
            @test isapprox(C[1,1], 1.0, atol = 0.01)
            @test isapprox(C[2,2], 1.0, atol = 0.01)
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # MSTGenerator — fit + sample
    # ════════════════════════════════════════════════════════════════════════
    @testset "MSTGenerator" begin
        rng_data = MersenneTwister(123)
        n = 100
        tbl = (;
            age    = rand(rng_data, 20:70, n),
            income = 30_000.0 .+ 50_000.0 .* rand(rng_data, n),
            gender = rand(rng_data, ["M", "F"], n),
            score  = randn(rng_data, n),
        )
        pb = PrivacyBudget(epsilon = 2.0, delta = 1e-5)

        @testset "basic fit + sample" begin
            model = fit(MSTGenerator(privacy = pb), tbl;
                        rng = MersenneTwister(42))
            @test model isa FittedMSTModel
            @test length(model.stat_columns) == 4
            @test model.n_original == n

            syn = sample(model, 80)
            @test syn isa NamedTuple
            @test length(syn.age) == 80
            @test length(syn.income) == 80
            @test length(syn.gender) == 80
            @test length(syn.score) == 80

            # Gender should only contain observed levels
            @test all(g -> g ∈ ["M", "F"], syn.gender)
        end

        @testset "DataFrame round-trip" begin
            df = DataFrame(tbl)
            model = fit(MSTGenerator(privacy = pb), df;
                        rng = MersenneTwister(42))
            syn = sample(model, 50)
            @test syn isa DataFrame
            @test nrow(syn) == 50
            @test Set(Symbol.(names(syn))) == Set([:age, :income, :gender, :score])
        end

        @testset "identifiers work with MST" begin
            df = DataFrame(
                id  = ["row_$i" for i in 1:n],
                val = randn(n),
                cat = rand(["a", "b", "c"], n),
            )
            model = fit(MSTGenerator(privacy = pb), df;
                        identifiers = [:id],
                        fill = Dict(:id => :sequential),
                        rng = MersenneTwister(42))
            syn = sample(model, 30)
            @test :id in Symbol.(names(syn))
            @test syn.id[1] == "id_1"
            @test !(:id in model.stat_columns)
        end

        # REQ-MST-005: belief propagation on the selected tree must be exact.
        # Checked against brute-force enumeration of the full joint, which is
        # feasible here because the state space is tiny (2*3*2 = 12 states).
        @testset "belief propagation is exact vs brute force" begin
            n_bins = [2, 3, 2]
            edges  = [(1, 2), (2, 3)]          # path 1 — 2 — 3, rooted at 1
            nbrs   = [[2], [1, 3], [2]]
            rng_bp = MersenneTwister(4)

            θ_node = [randn(rng_bp, k) for k in n_bins]
            θ_edge = Dict{Tuple{Int,Int}, Matrix{Float64}}(
                (1, 2) => randn(rng_bp, 2, 3),
                (2, 3) => randn(rng_bp, 3, 2))

            joint = zeros(n_bins...)
            for a in 1:2, b in 1:3, c in 1:2
                joint[a, b, c] = exp(θ_node[1][a] + θ_node[2][b] + θ_node[3][c] +
                                     θ_edge[(1, 2)][a, b] + θ_edge[(2, 3)][b, c])
            end
            joint ./= sum(joint)

            μ_node, μ_edge = DataMimic._tree_bp(edges, nbrs, 1, n_bins,
                                                θ_node, θ_edge)

            @test μ_node[1] ≈ vec(sum(joint, dims = (2, 3)))
            @test μ_node[2] ≈ vec(sum(joint, dims = (1, 3)))
            @test μ_node[3] ≈ vec(sum(joint, dims = (1, 2)))
            @test μ_edge[(1, 2)] ≈ dropdims(sum(joint, dims = 3), dims = 3)
            @test μ_edge[(2, 3)] ≈ dropdims(sum(joint, dims = 1), dims = 1)

            @test all(isapprox(sum(m), 1.0) for m in μ_node)
            @test vec(sum(μ_edge[(1, 2)], dims = 1)) ≈ μ_node[2]
            @test vec(sum(μ_edge[(2, 3)], dims = 2)) ≈ μ_node[2]
        end

        # The estimation step exists to reconcile inconsistent measurements:
        # given consistent targets it should recover them, and given
        # inconsistent ones it must still return mutually consistent marginals.
        @testset "mirror descent reconciles noisy marginals" begin
            n_bins = [3, 2]
            edges  = [(1, 2)]
            nbrs   = [[2], [1]]

            truth = [0.30 0.10;
                     0.05 0.25;
                     0.20 0.10]
            y_node = [vec(sum(truth, dims = 2)), vec(sum(truth, dims = 1))]
            y_edge = Dict{Tuple{Int,Int}, Matrix{Float64}}((1, 2) => truth)

            μn, μe = DataMimic._fit_tree_mrf(edges, nbrs, 1, n_bins,
                                             y_node, y_edge; iters = 400)
            @test μe[(1, 2)] ≈ truth atol = 1e-3
            @test μn[1] ≈ y_node[1] atol = 1e-3

            # Perturb the 1-way target so it disagrees with the 2-way one; the
            # fit must still return consistent marginals.
            bad_node = [[0.6, 0.2, 0.2], y_node[2]]
            μn2, μe2 = DataMimic._fit_tree_mrf(edges, nbrs, 1, n_bins,
                                               bad_node, y_edge; iters = 400)
            @test vec(sum(μe2[(1, 2)], dims = 2)) ≈ μn2[1] atol = 1e-8
            @test vec(sum(μe2[(1, 2)], dims = 1)) ≈ μn2[2] atol = 1e-8
            @test isapprox(sum(μe2[(1, 2)]), 1.0; atol = 1e-8)
        end

        @testset "reproducibility" begin
            m1 = fit(MSTGenerator(privacy = pb), tbl; rng = MersenneTwister(1))
            m2 = fit(MSTGenerator(privacy = pb), tbl; rng = MersenneTwister(1))

            s1 = sample(m1, 40; rng = MersenneTwister(99))
            s2 = sample(m2, 40; rng = MersenneTwister(99))
            @test s1.gender == s2.gender
            @test s1.age    == s2.age
        end

        @testset "single column" begin
            single = (; x = rand(rng_data, ["a", "b", "c"], 50))
            model = fit(MSTGenerator(privacy = pb), single;
                        rng = MersenneTwister(42))
            @test isempty(model.tree_edges)
            syn = sample(model, 20)
            @test length(syn.x) == 20
            @test all(v -> v ∈ ["a", "b", "c"], syn.x)
        end

        @testset "all categorical data" begin
            cat_tbl = (;
                a = rand(rng_data, ["x", "y", "z"], n),
                b = rand(rng_data, ["p", "q"], n),
            )
            model = fit(MSTGenerator(privacy = pb), cat_tbl;
                        rng = MersenneTwister(42))
            syn = sample(model, 40)
            @test all(v -> v ∈ ["x", "y", "z"], syn.a)
            @test all(v -> v ∈ ["p", "q"], syn.b)
        end
    end

    @testset "MST domain compression" begin
        using DataMimic: _compress_domain, _identity_compression, is_identity,
                         _expand_vector, _expand_conditional,
                         MST_DOMAIN_COMPRESSION, MST_COMPRESSION_STATS

        @testset "grouping" begin
            # Two dense bins kept apart, three sparse ones folded together.
            c = _compress_domain([100.0, 90.0, 1.0, 2.0, 0.5], 10.0)
            @test !is_identity(c)
            @test c.groups == [[1], [2], [3, 4, 5]]
            @test c.map == [1, 2, 3, 3, 3]

            # One bin below the line is not worth a merge: a group of one is
            # the bin itself, so nothing is gained and the domain is untouched.
            @test is_identity(_compress_domain([100.0, 90.0, 1.0], 10.0))

            # Nothing above the line: collapsing every bin into one would
            # delete the column rather than denoise it.
            @test is_identity(_compress_domain([1.0, 2.0, 3.0], 10.0))

            # Noise can push a count negative; that bin is simply sparse.
            c2 = _compress_domain([50.0, -3.0, -1.0], 10.0)
            @test c2.groups == [[1], [2, 3]]
        end

        @testset "expansion preserves normalization" begin
            c = _compress_domain([100.0, 90.0, 1.0, 2.0, 0.5], 10.0)

            v = _expand_vector(c, [0.5, 0.2, 0.3])
            @test length(v) == 5
            @test sum(v) ≈ 1.0
            # Merged members are indistinguishable to the model, so uniform.
            @test v[3] ≈ v[4] ≈ v[5] ≈ 0.1

            ident = _identity_compression(3)
            @test _expand_vector(ident, [0.2, 0.3, 0.5]) == [0.2, 0.3, 0.5]

            cond = [0.6 0.1 0.3; 0.2 0.5 0.3; 0.1 0.1 0.8]
            full = _expand_conditional(c, c, cond)
            @test size(full) == (5, 5)
            @test all(isapprox.(sum(full; dims = 2), 1.0))
            # Rows of merged parents are copies: the model cannot tell them apart.
            @test full[3, :] == full[4, :] == full[5, :]
        end

        @testset "end to end" begin
            # A long Zipf tail is what compression exists for.
            rng = MersenneTwister(4)
            w = [1.0 / l for l in 1:120]; w ./= sum(w); cum = cumsum(w)
            df = DataFrame(num = randn(rng, 3_000))
            for j in 1:4
                df[!, Symbol("c", j)] =
                    [string(searchsortedfirst(cum, rand(rng))) for _ in 1:3_000]
            end
            pb = PrivacyBudget(epsilon = 1.0, delta = 1e-5)

            m = fit(MSTGenerator(privacy = pb), df; rng = MersenneTwister(5))
            st = MST_COMPRESSION_STATS[]
            @test st.bins_after < st.bins_before      # something actually merged

            # Everything stored must be on the ORIGINAL domain: compression is
            # an estimation-time device and must not leak into the model.
            nb = Dict(i => m.discretization[c].n_bins
                      for (i, c) in enumerate(m.stat_columns))
            @test length(m.root_marginal) == nb[m.root]
            @test all(size(m.conditionals[(p, c)]) == (nb[p], nb[c])
                      for (p, c) in m.tree_edges)
            @test sum(m.root_marginal) ≈ 1.0
            @test all(all(isapprox.(sum(c; dims = 2), 1.0))
                      for c in values(m.conditionals))

            syn = sample(m, 500; rng = MersenneTwister(6))
            @test nrow(syn) == 500
            @test isequal(syn, sample(m, 500; rng = MersenneTwister(6)))
            for c in names(df)
                eltype(df[!, c]) <: AbstractString || continue
                @test issubset(Set(syn[!, c]), Set(df[!, c]))
            end

            # The toggle must be a true no-op, so the benchmark's off arm
            # really is the uncompressed mechanism.
            prev = MST_DOMAIN_COMPRESSION[]
            MST_DOMAIN_COMPRESSION[] = false
            try
                a = fit(MSTGenerator(privacy = pb), df; rng = MersenneTwister(5))
                b = fit(MSTGenerator(privacy = pb), df; rng = MersenneTwister(5))
                @test a.tree_edges == b.tree_edges
                @test a.root_marginal == b.root_marginal
                @test MST_COMPRESSION_STATS[].bins_after ==
                      MST_COMPRESSION_STATS[].bins_before
            finally
                MST_DOMAIN_COMPRESSION[] = prev
            end
            @test MST_DOMAIN_COMPRESSION[]            # restored, and on by default
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # DPCopulaGenerator — fit + sample
    # ════════════════════════════════════════════════════════════════════════
    @testset "DPCopulaGenerator" begin
        rng_data = MersenneTwister(456)
        n = 100
        tbl = (;
            x   = randn(rng_data, n),
            y   = randn(rng_data, n) .* 2 .+ 1,
            cat = rand(rng_data, ["A", "B", "C"], n),
        )
        pb = PrivacyBudget(epsilon = 4.0, delta = 1e-5)

        @testset "basic fit + sample" begin
            model = fit(DPCopulaGenerator(privacy = pb), tbl;
                        rng = MersenneTwister(42))
            @test model isa FittedDPCopulaModel
            @test :x in model.copula_columns
            @test :y in model.copula_columns
            @test model.n_original == n
            @test model.copula !== nothing              # GaussianCopula built

            syn = sample(model, 60)
            @test syn isa NamedTuple
            @test length(syn.x) == 60
            @test length(syn.y) == 60
            @test length(syn.cat) == 60
            @test all(c -> c ∈ ["A", "B", "C"], syn.cat)
        end

        @testset "DataFrame round-trip" begin
            df = DataFrame(tbl)
            model = fit(DPCopulaGenerator(privacy = pb), df;
                        rng = MersenneTwister(42))
            syn = sample(model, 50)
            @test syn isa DataFrame
            @test nrow(syn) == 50
        end

        @testset "single numeric column" begin
            single = (;
                x   = randn(rng_data, 50),
                cat = rand(rng_data, ["a", "b"], 50),
            )
            model = @test_logs (:warn, r"one numeric") fit(
                DPCopulaGenerator(privacy = pb), single;
                rng = MersenneTwister(42))
            @test isnothing(model.copula)
            syn = sample(model, 20)
            @test length(syn.x) == 20
        end

        @testset "all categorical" begin
            cat_tbl = (;
                a = rand(rng_data, ["x", "y", "z"], 50),
                b = rand(rng_data, ["p", "q"], 50),
            )
            model = @test_logs (:warn, r"No numeric") fit(
                DPCopulaGenerator(privacy = pb), cat_tbl;
                rng = MersenneTwister(42))
            syn = sample(model, 20)
            @test all(v -> v ∈ ["x", "y", "z"], syn.a)
        end

        # REQ-DPC-002: Analyze-Gauss requires a *symmetric* noise matrix whose
        # entries each carry the calibrated variance.  Building it as
        # (E + E')/2 from independent draws is the natural-looking mistake: it
        # leaves the diagonal at σ² but halves every off-diagonal to σ²/2,
        # under-noising them by √2 and breaking the privacy calibration.
        @testset "Analyze-Gauss noise is symmetric with uniform variance" begin
            d, reps, σ = 4, 4000, 2.0
            rng_noise = MersenneTwister(7)
            draw_E() = DataMimic._symmetric_gaussian_noise(d, σ, rng_noise)

            @test all(E == E' for E in (draw_E() for _ in 1:20))

            diag_draws = Float64[]
            off_draws  = Float64[]
            for _ in 1:reps
                E = draw_E()
                push!(diag_draws, E[1, 1])
                push!(off_draws,  E[1, 2])
            end

            # Both must have variance σ², not σ²/2 for the off-diagonal.
            svar = DataMimic.StatsBase.var
            @test isapprox(svar(diag_draws), σ^2; rtol = 0.1)
            @test isapprox(svar(off_draws),  σ^2; rtol = 0.1)

            # Guard against the averaging bug specifically: it would land the
            # off-diagonal variance near σ²/2.
            @test !isapprox(svar(off_draws), σ^2 / 2; rtol = 0.1)
        end

        @testset "reproducibility" begin
            m1 = fit(DPCopulaGenerator(privacy = pb), tbl; 
                     rng = MersenneTwister(1))
            m2 = fit(DPCopulaGenerator(privacy = pb), tbl; 
                     rng = MersenneTwister(1))

            s1 = sample(m1, 40; rng = MersenneTwister(99))
            s2 = sample(m2, 40; rng = MersenneTwister(99))
            @test s1.x   == s2.x
            @test s1.cat == s2.cat
        end

        @testset "identifiers work with DPCopula" begin
            df = DataFrame(
                id  = ["row_$i" for i in 1:n],
                val = randn(rng_data, n),
                cat = rand(rng_data, ["a", "b"], n),
            )
            model = fit(DPCopulaGenerator(privacy = pb), df;
                        identifiers = [:id],
                        fill = Dict(:id => :sequential_int),
                        rng = MersenneTwister(42))
            syn = sample(model, 25)
            @test syn.id == collect(1:25)
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2 missingness
    # ════════════════════════════════════════════════════════════════════════
    @testset "missingness with private generators" begin
        n = 200
        col = Vector{Union{Float64, Missing}}(randn(n))
        col[1:20] .= missing   # 10% missing
        tbl = (; x = col, y = randn(n))
        pb = PrivacyBudget(epsilon = 4.0)

        @testset "MST" begin
            model = fit(MSTGenerator(privacy = pb), tbl; 
                        rng = MersenneTwister(42))
            @test isapprox(model.missingness[:x], 0.1, atol = 0.01)
            syn = sample(model, 500)
            p_miss = count(ismissing, syn.x) / 500
            @test p_miss > 0.0
            @test Missing <: eltype(syn.x)
        end

        @testset "DPCopula" begin
            model = fit(DPCopulaGenerator(privacy = pb), tbl; 
                        rng = MersenneTwister(42))
            @test isapprox(model.missingness[:x], 0.1, atol = 0.01)
            syn = sample(model, 500)
            p_miss = count(ismissing, syn.x) / 500
            @test p_miss > 0.0
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2 serialization
    # ════════════════════════════════════════════════════════════════════════
    @testset "save / load (Phase 2)" begin
        pb = PrivacyBudget(epsilon = 2.0)
        tbl = (; x = randn(50), y = rand(["a", "b"], 50))

        @testset "MST model" begin
            model = fit(MSTGenerator(privacy = pb), tbl; 
                        rng = MersenneTwister(42))
            path = tempname() * ".dmimic"
            save(path, model)
            loaded = load(path)
            @test loaded isa FittedMSTModel
            @test loaded.column_names == model.column_names
            syn = sample(loaded, 20)
            @test length(syn.x) == 20
            rm(path)
        end

        @testset "DPCopula model" begin
            model = fit(DPCopulaGenerator(privacy = pb), tbl; 
                        rng = MersenneTwister(42))
            path = tempname() * ".dmimic"
            save(path, model)
            loaded = load(path)
            @test loaded isa FittedDPCopulaModel
            syn = sample(loaded, 20)
            @test length(syn.x) == 20
            rm(path)
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2 error conditions
    # ════════════════════════════════════════════════════════════════════════
    @testset "Phase 2 error conditions" begin
        pb = PrivacyBudget(epsilon = 1.0)
        tbl = (; x = randn(50), y = rand(["a", "b"], 50))

        # A private generator without a budget is now unconstructable, so
        # there is no fit-time rejection left to test.
        @test_throws ArgumentError MSTGenerator()
        @test_throws ArgumentError DPCopulaGenerator()

        # sample with n < 1
        model = fit(MSTGenerator(privacy = pb), tbl; 
                    rng = MersenneTwister(42))
        @test_throws ArgumentError sample(model, 0)

        model2 = fit(DPCopulaGenerator(privacy = pb), tbl; 
                     rng = MersenneTwister(42))
        @test_throws ArgumentError sample(model2, 0)

        # DiffusionGenerator: the budget's presence is what selects DP-SGD,
        # so there is no `dp` flag left that could disagree with it, and
        # nothing for `fit` to validate.
        @test privacy_budget(DiffusionGenerator()) === nothing
        @test privacy_budget(DiffusionGenerator(; privacy = pb)) === pb
    end

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2 synthesize convenience
    # ════════════════════════════════════════════════════════════════════════
    @testset "synthesize with private generators" begin
        pb = PrivacyBudget(epsilon = 2.0)
        df = DataFrame(x = randn(80), y = rand(["a", "b", "c"], 80))

        syn = synthesize(MSTGenerator(privacy = pb), df, 40;
                         rng = MersenneTwister(42))
        @test syn isa DataFrame
        @test nrow(syn) == 40

        syn2 = synthesize(DPCopulaGenerator(privacy = pb), df, 40;
                          rng = MersenneTwister(42))
        @test syn2 isa DataFrame
        @test nrow(syn2) == 40
    end

    # ════════════════════════════════════════════════════════════════════════
    #  P H A S E   3  —  D I F F U S I O N   G E N E R A T O R
    # ════════════════════════════════════════════════════════════════════════

    @testset "DiffusionGenerator" begin
        rng_data = MersenneTwister(789)
        n = 80

        @testset "mixed data (numeric + categorical)" begin
            tbl = (;
                x   = randn(rng_data, Float32, n),
                y   = randn(rng_data, Float32, n) .* 2f0 .+ 1f0,
                cat = rand(rng_data, ["a", "b", "c"], n),
            )

            model = fit(DiffusionGenerator(; epochs = 3, batch_size = 16),
                        tbl; rng = MersenneTwister(42))
            @test model isa FittedDiffusionModel
            @test model.n_original == n
            @test :x in model.column_names
            @test :y in model.column_names
            @test :cat in model.column_names
            @test length(model.num_columns) == 2
            @test length(model.cat_columns) == 1

            syn = sample(model, 30)
            @test syn isa NamedTuple
            @test length(syn.x) == 30
            @test length(syn.y) == 30
            @test length(syn.cat) == 30
            @test all(c -> c ∈ ["a", "b", "c"], syn.cat)
        end

        @testset "numeric-only data" begin
            tbl = (;
                a = randn(rng_data, Float32, n),
                b = randn(rng_data, Float32, n),
            )
            model = fit(DiffusionGenerator(; epochs = 2, batch_size = 32),
                        tbl; rng = MersenneTwister(42))
            @test isempty(model.cat_columns)
            @test length(model.num_columns) == 2

            syn = sample(model, 20)
            @test length(syn.a) == 20
            @test eltype(syn.a) <: AbstractFloat
        end

        @testset "categorical-only data" begin
            tbl = (;
                c1 = rand(rng_data, ["x", "y", "z"], n),
                c2 = rand(rng_data, ["p", "q"], n),
            )
            model = fit(DiffusionGenerator(; epochs = 2, batch_size = 32),
                        tbl; rng = MersenneTwister(42))
            @test isempty(model.num_columns)
            @test length(model.cat_columns) == 2

            syn = sample(model, 20)
            @test length(syn.c1) == 20
            @test all(v -> v ∈ ["x", "y", "z"], syn.c1)
            @test all(v -> v ∈ ["p", "q"], syn.c2)
        end

        @testset "DataFrame round-trip" begin
            df = DataFrame(
                x   = randn(rng_data, Float32, n),
                cat = rand(rng_data, ["a", "b"], n),
            )
            model = fit(DiffusionGenerator(; epochs = 2, batch_size = 32),
                        df; rng = MersenneTwister(42))
            syn = sample(model, 25)
            @test syn isa DataFrame
            @test nrow(syn) == 25
        end

        @testset "identifiers" begin
            tbl = (;
                id  = ["row_$i" for i in 1:n],
                val = randn(rng_data, Float32, n),
                cat = rand(rng_data, ["a", "b", "c"], n),
            )
            model = fit(DiffusionGenerator(; epochs = 2, batch_size = 32),
                        tbl; identifiers = [:id],
                        fill = Dict(:id => :sequential_int),
                        rng = MersenneTwister(42))
            @test :id in model.identifier_columns
            syn = sample(model, 15)
            @test syn.id == collect(1:15)
        end

        @testset "missingness" begin
            col = Vector{Union{Float32, Missing}}(randn(rng_data, Float32, n))
            col[1:8] .= missing   # 10% missing
            tbl = (; x = col, y = randn(rng_data, Float32, n))
            model = fit(DiffusionGenerator(; epochs = 2, batch_size = 32),
                        tbl; rng = MersenneTwister(42))
            @test isapprox(model.missingness[:x], 0.1, atol = 0.02)
            syn = sample(model, 200)
            @test Missing <: eltype(syn.x)
        end

        @testset "sample with n < 1 → error" begin
            tbl = (;
                x = randn(rng_data, Float32, n),
                y = rand(rng_data, ["a", "b"], n),
            )
            model = fit(DiffusionGenerator(; epochs = 2, batch_size = 32),
                        tbl; rng = MersenneTwister(42))
            @test_throws ArgumentError sample(model, 0)
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # DiffusionGenerator — DP-SGD (private)
    # ════════════════════════════════════════════════════════════════════════
    @testset "DiffusionGenerator DP-SGD" begin
        rng_data = MersenneTwister(101)
        n = 80
        tbl = (;
            x   = randn(rng_data, Float32, n),
            y   = randn(rng_data, Float32, n),
            cat = rand(rng_data, ["a", "b", "c"], n),
        )
        pb = PrivacyBudget(epsilon = 10.0, delta = 1e-5)

        @testset "fit + sample under a privacy budget" begin
            model = fit(DiffusionGenerator(; epochs = 2,
                                             batch_size = 16),
                        tbl; rng = MersenneTwister(42))
            @test model isa FittedDiffusionModel
            syn = sample(model, 20)
            @test length(syn.x) == 20
            @test all(c -> c ∈ ["a", "b", "c"], syn.cat)
        end

        # A diverged run used to report loss=NaN and keep going: gradients
        # NaN, every weight NaN, remaining epochs wasted, and nothing surfaced
        # until sampling failed much later with an unrelated-looking error.
        @testset "training aborts on non-finite loss" begin
            ext = Base.get_extension(DataMimic, :DataMimicLuxExt)
            @test ext._check_finite_loss(1.5, 3, 100, 0.001) === nothing
            for bad in (NaN, Inf, -Inf)
                err = try
                    ext._check_finite_loss(bad, 7, 100, 0.002); nothing
                catch e
                    sprint(showerror, e)
                end
                @test err !== nothing
                @test occursin("diverged", err)
                @test occursin("epoch 7", err)   # names where it happened
                @test occursin("lr", err) || occursin("learning rate", err)
            end
        end

        # REQ-DIF-006: the accountant models the *Poisson*-subsampled
        # Gaussian mechanism.  At q = 1 that degenerates to the plain
        # Gaussian mechanism, whose RDP is exactly α/(2σ²) — a closed form
        # the implementation must reproduce.
        @testset "RDP accountant reduces to the Gaussian mechanism at q=1" begin
            ext = Base.get_extension(DataMimic, :DataMimicLuxExt)
            σ, δ = 2.0, 1e-5
            alphas = vcat(collect(2:10), collect(12:2:64), [128, 256])
            analytic = minimum(a / (2σ^2) + log(1 / δ) / (a - 1) for a in alphas)
            @test ext._rdp_accountant(σ, 1.0, 1, δ) ≈ analytic

            # ε must grow with the sampling rate and the number of steps,
            # and shrink as noise is added.
            @test issorted([ext._rdp_accountant(s, 0.01, 1000, δ)
                            for s in (0.5, 1.0, 2.0, 4.0)]; rev = true)
            @test issorted([ext._rdp_accountant(1.0, q, 1000, δ)
                            for q in (0.001, 0.01, 0.1, 0.5)])
            @test issorted([ext._rdp_accountant(1.0, 0.01, t, δ)
                            for t in (10, 100, 1000, 10_000)])
        end

        # REQ-DIF-005: Poisson subsampling produces variable lot sizes, and
        # an empty lot is a legitimate outcome that must still take a noisy
        # step.  n=8 with batch_size=1 gives q=0.125, so P(empty) ≈ 0.34 per
        # step — over 16 steps an empty lot is effectively certain.
        @testset "Poisson subsampling tolerates empty lots" begin
            small = (; x = randn(MersenneTwister(7), Float32, 8),
                       c = rand(MersenneTwister(8), ["a", "b"], 8))
            model = fit(DiffusionGenerator(; epochs = 2,
                                             batch_size = 1, num_timesteps = 10,
                                             hidden_dim = 8, n_blocks = 1),
                        small; rng = MersenneTwister(42))
            @test model isa FittedDiffusionModel
            syn = sample(model, 5)
            @test length(syn.x) == 5
            @test all(c -> c ∈ ["a", "b"], syn.c)
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # DiffusionGenerator serialization
    # ════════════════════════════════════════════════════════════════════════
    @testset "save / load (Diffusion)" begin
        rng_data = MersenneTwister(202)
        tbl = (;
            x   = randn(rng_data, Float32, 60),
            cat = rand(rng_data, ["a", "b"], 60),
        )
        model = fit(DiffusionGenerator(; epochs = 2, batch_size = 32),
                    tbl; rng = MersenneTwister(42))
        path = tempname() * ".dmimic"
        save(path, model)
        loaded = load(path)
        @test loaded isa FittedDiffusionModel
        @test loaded.column_names == model.column_names
        @test loaded.n_original == model.n_original

        syn = sample(loaded, 15)
        @test length(syn.x) == 15
        rm(path)
    end

    # ════════════════════════════════════════════════════════════════════════
    # DiffusionGenerator on a wide table
    # ════════════════════════════════════════════════════════════════════════
    @testset "DiffusionGenerator wide table" begin
        # 35 numeric columns: more features than the default block width is
        # tuned for, which is where shape bugs in the denoiser surface.
        cols_nt = NamedTuple{Tuple(Symbol.("c" .* string.(1:35)))}(
            Tuple(randn(Float32, 60) for _ in 1:35))
        model = fit(DiffusionGenerator(epochs = 2), cols_nt;
                    rng = MersenneTwister(42))
        @test model isa FittedDiffusionModel
    end

    # ════════════════════════════════════════════════════════════════════════
    # Identifier detection needs enough observations to mean anything
    # ════════════════════════════════════════════════════════════════════════
    #
    # The heuristic divided unique values by NON-MISSING values with no floor,
    # so a column with a single observation scored a ratio of 1.0 and was
    # classified as an identifier - then silently dropped from the output.
    # Found by an OpenML sweep on the `anneal` dataset, whose :bc column has
    # 898 rows and exactly one non-missing value.
    @testset "identifier detection needs observations" begin
        n = 400
        rng = MersenneTwister(1)

        # One observation in a sea of missing: a constant, not a key.
        sparse_col = Vector{Union{Missing,String}}(missing, n)
        sparse_col[7] = "Y"

        # A handful of distinct observations is still not a key.
        few_col = Vector{Union{Missing,String}}(missing, n)
        for (i, v) in enumerate(["a", "b", "c", "d", "e"])
            few_col[i * 10] = v
        end

        df = DataFrame(num = randn(rng, n),
                       cat = rand(rng, ["x", "y", "z"], n),
                       sparse_one = sparse_col,
                       sparse_few = few_col)

        m = fit(CopulaGenerator(), df; rng = MersenneTwister(2))
        syn = sample(m, 100; rng = MersenneTwister(3))

        # Neither should be treated as an identifier, so neither is dropped.
        @test :sparse_one in Symbol.(names(syn))
        @test :sparse_few in Symbol.(names(syn))

        # A genuine identifier - many observations, all distinct - still is one.
        df2 = DataFrame(num = randn(rng, n),
                        cat = rand(rng, ["x", "y"], n),
                        key = ["id_" * string(i) for i in 1:n])
        m2 = fit(CopulaGenerator(), df2; rng = MersenneTwister(4))
        syn2 = sample(m2, 100; rng = MersenneTwister(5))
        @test !(:key in Symbol.(names(syn2)))
    end

    # ════════════════════════════════════════════════════════════════════════
    # Hinting an integer column as continuous
    # ════════════════════════════════════════════════════════════════════════
    #
    # `_cast_numeric` converted sampled values back to the column's original
    # eltype. For a column stored as Int64 but hinted :continuous, that meant
    # convert(Vector{Int64}, [13.28, ...]) and an InexactError. The hint is the
    # caller explicitly asking for continuous modelling, so continuous values
    # are what should come back.
    @testset "integer column hinted continuous" begin
        df = DataFrame(a = rand(MersenneTwister(1), 1:16, 500),
                       b = randn(MersenneTwister(2), 500))

        m = fit(CopulaGenerator(), df;
                hints = [ColumnHint(name = :a, kind = :continuous)],
                rng = MersenneTwister(3))
        s = sample(m, 200; rng = MersenneTwister(4))

        @test eltype(s.a) <: AbstractFloat
        @test all(isfinite, s.a)
        # Continuous means continuous: at least some draws are not whole.
        @test any(v -> abs(v - round(v)) > 1e-9, s.a)

        # The other kinds still round-trip to integers.
        for kind in (:integer, :categorical)
            mk = fit(CopulaGenerator(), df;
                     hints = [ColumnHint(name = :a, kind = kind)],
                     rng = MersenneTwister(3))
            sk = sample(mk, 200; rng = MersenneTwister(4))
            @test eltype(sk.a) <: Integer
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Learning-rate annealing
    # ════════════════════════════════════════════════════════════════════════
    #
    # The reference counts steps from zero, so lr runs from lr_max down to
    # lr_max/total and never hits zero. Using the 1-based epoch directly made
    # the last epoch of every run train at exactly 0, and `epochs = 1` a
    # complete no-op.
    @testset "anneal_lr" begin
        Ext = Base.get_extension(DataMimic, :DataMimicLuxExt)
        lr = 1e-3

        @test Ext._anneal_lr(1, 1, lr, 0) > 0            # epochs=1 must train
        @test Ext._anneal_lr(1, 10, lr, 0) ≈ lr          # starts at lr_max
        @test Ext._anneal_lr(10, 10, lr, 0) ≈ lr / 10    # ends at lr_max/total
        for E in (1, 2, 5, 20)
            sched = [Ext._anneal_lr(e, E, lr, 0) for e in 1:E]
            @test all(>(0), sched)
            @test issorted(sched; rev = true)
            @test maximum(sched) ≈ lr
        end
        # Warmup ramps up before the anneal, and never goes negative.
        wsched = [Ext._anneal_lr(e, 6, lr, 2) for e in 1:6]
        @test wsched[1] < wsched[2]
        @test all(>=(0), wsched)
    end

    # ════════════════════════════════════════════════════════════════════════
    # Ghost clipping equals the per-example loop
    # ════════════════════════════════════════════════════════════════════════
    #
    # DP-SGD's fast path computes per-example gradient norms and the clipped
    # gradient sum from column norms and two matmuls, instead of one backward
    # pass per example. That is an algebraic identity, so the two must agree to
    # floating-point rounding — and if they ever stop agreeing, the privacy
    # guarantee is what breaks, silently. `_ghost_forward` mirrors
    # `TabDDPMBackbone` by hand, so this is also what catches a change to one
    # without the other.
    @testset "ghost clipping matches per-example gradients" begin
        Ext = Base.get_extension(DataMimic, :DataMimicLuxExt)
        dev = Lux.cpu_device()

        function worst_reldiff(a, b)
            worst = 0.0
            function walk(x, y)
                if x isa NamedTuple
                    for k in keys(x); walk(getfield(x, k), getfield(y, k)); end
                elseif x isa AbstractArray
                    sc = max(maximum(abs, x), maximum(abs, y), 1e-8)
                    worst = max(worst, maximum(abs.(x .- y)) / sc)
                end
            end
            walk(a, b)
            worst
        end

        function compare_paths(; d_num, cat_dims, bs, d_layers, n_classes, T = 50, seed = 1)
            rng = MersenneTwister(seed)
            d_cat = sum(cat_dims; init = 0)
            plan  = isempty(cat_dims) ? nothing : Ext._block_plan(cat_dims, dev)
            log_K = isempty(cat_dims) ? Float32[] : Ext._log_K_vector(cat_dims)
            betas, _ = Ext._cosine_schedule(T)
            sched = Ext._schedule_constants(betas)
            lca_T = Float32(sched.log_cumprod_alpha[T])
            l1m_T = Float32(sched.log_1_min_cumprod_alpha[T])

            backbone, _ = Ext._build_model(d_num + d_cat, d_num, cat_dims;
                d_layers = d_layers, embed_dim = 32, dropout = 0.0,
                n_classes = n_classes)
            emb_layer = Ext.SinusoidalEmbedding(32)
            ps_bb, st_bb = Lux.setup(rng, backbone)
            ps_em, st_em = Lux.setup(rng, emb_layer)
            ps_all = (; backbone = ps_bb, emb = ps_em)

            x_oh = zeros(Float32, d_cat, bs)
            if d_cat > 0
                off = 0
                for K in cat_dims
                    for b in 1:bs; x_oh[off + rand(rng, 1:K), b] = 1f0; end
                    off += K
                end
            end
            log_x0 = d_cat > 0 ? Ext._to_log_onehot(x_oh) : zeros(Float32, 0, bs)
            tvec   = rand(rng, 1:T, bs)
            coef_b = Ext._batch_coefs(Ext._device_schedule(sched, dev), tvec)
            log_xt = d_cat > 0 ?
                Ext._multinomial_q_sample(log_x0, plan, coef_b, log_K, rng, dev) :
                zeros(Float32, 0, bs)
            xnum = randn(rng, Float32, d_num, bs)
            eps  = randn(rng, Float32, d_num, bs)
            yidx = n_classes > 0 ? rand(rng, 1:n_classes, bs) : Int[]
            C = 1.0

            ref, _ = Ext._dpsgd_grads_loop(backbone, emb_layer, ps_all, st_bb, st_em,
                        xnum, log_xt, log_x0, eps, tvec, collect(1:bs), yidx,
                        n_classes > 0, sched, d_num, plan, T, log_K, lca_T, l1m_T,
                        C, dev, bs)

            t_emb, _ = Lux.apply(emb_layer, Float32.(tvec .- 1), ps_em, st_em)
            features = d_num > 0 && d_cat > 0 ? vcat(xnum, log_xt) :
                       (d_num > 0 ? xnum : log_xt)
            y_dev = n_classes > 0 ? yidx : nothing
            out, cache = Ext._ghost_forward(backbone, ps_bb, features, t_emb, y_dev)
            _, dout = Ext._compute_output_grad(Ext.AD_BACKEND, out) do o
                sum(Ext._diffusion_loss_vec(o, log_xt, log_x0, coef_b, eps,
                                            d_num, plan, T, log_K, lca_T, l1m_T))
            end
            sq, build = Ext._ghost_backward(backbone, ps_bb, cache, dout)
            cvec = min.(1f0, Float32(C) ./ max.(sqrt.(sq), 1f-12))
            return worst_reldiff(build(cvec), ref.backbone)
        end

        @test compare_paths(d_num = 4, cat_dims = Int[],   bs = 24, d_layers = [16,16], n_classes = 0) < 1e-4
        @test compare_paths(d_num = 0, cat_dims = [3,5,4], bs = 24, d_layers = [16,16], n_classes = 0) < 1e-4
        @test compare_paths(d_num = 4, cat_dims = [3,5,4], bs = 24, d_layers = [16,16], n_classes = 0) < 1e-4
        @test compare_paths(d_num = 4, cat_dims = [3,5,4], bs = 24, d_layers = [32,16,8], n_classes = 0) < 1e-4
        @test compare_paths(d_num = 3, cat_dims = [4,2],   bs = 24, d_layers = [16,16], n_classes = 3) < 1e-4
        @test compare_paths(d_num = 4, cat_dims = [3,5],   bs = 24, d_layers = [16],    n_classes = 0) < 1e-4
        @test compare_paths(d_num = 2, cat_dims = [9,7,6,5,2], bs = 32, d_layers = [16,16], n_classes = 0) < 1e-4

        # And the whole training loop agrees, not just one step.
        @testset "end to end" begin
            n = 400; d_num = 2; cat_dims = [4, 3]; d_cat = sum(cat_dims); T = 30
            r0 = MersenneTwister(9)
            X_num = randn(r0, Float32, d_num, n)
            X_cat = zeros(Float32, d_cat, n)
            off = 0
            for K in cat_dims
                for b in 1:n; X_cat[off + rand(r0, 1:K), b] = 1f0; end
                off += K
            end
            betas, _ = Ext._cosine_schedule(T)
            sched = Ext._schedule_constants(betas)

            function trained(force_loop)
                r = MersenneTwister(9)
                bb, _ = Ext._build_model(d_num + d_cat, d_num, cat_dims;
                            d_layers = [32, 32], embed_dim = 16, dropout = 0.0,
                            n_classes = 0)
                em = Ext.SinusoidalEmbedding(16)
                sr = MersenneTwister(9)
                pb, sb = Lux.setup(sr, bb); pe, se = Lux.setup(sr, em)
                Ext._train_dpsgd!(bb, em, pb, pe, sb, se, X_num, X_cat, Int[],
                                  sched, cat_dims, d_num, 3, 128, 1e-3, 0, 1e-4,
                                  PrivacyBudget(epsilon = 10.0), r, dev;
                                  force_loop = force_loop)
            end
            @test worst_reldiff(trained(false)[1], trained(true)[1]) < 1e-4
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Per-block log-softmax stability
    # ════════════════════════════════════════════════════════════════════════
    #
    # `_block_log_normalize` once subtracted a single GLOBAL maximum, on the
    # assumption that its inputs are log-probabilities. `_predict_start` calls
    # it on raw network logits, which at initialization span roughly ±100, so
    # a block sitting far below the global maximum underflowed to zero across
    # all its rows: block sum 0, log(0) = -Inf, z - (-Inf) = +Inf, and every
    # downstream KL became NaN.
    #
    # It needs several blocks AND a wide batch to appear — a single block is
    # always safe, because then the global maximum is the block maximum.
    @testset "block log-softmax stability" begin
        Ext = Base.get_extension(DataMimic, :DataMimicLuxExt)
        dev = Lux.cpu_device()

        # Reference: the obvious per-block log-softmax.
        function ref_lognorm(x, dims)
            out = similar(x); off = 0
            for K in dims
                blk = @view x[off+1:off+K, :]
                m = maximum(blk; dims = 1)
                z = blk .- m
                out[off+1:off+K, :] = z .- log.(sum(exp.(z); dims = 1))
                off += K
            end
            out
        end

        for dims in ([3], [12, 12, 12], [9, 16, 7, 15, 6, 5, 2])
            plan = Ext._block_plan(dims, dev)
            for sd in (1f0, 40f0, 100f0)
                x = randn(MersenneTwister(3), Float32, sum(dims), 512) .* sd
                got = Ext._block_log_normalize(x, plan)
                @test all(isfinite, got)
                @test maximum(abs.(got .- ref_lognorm(x, dims))) < 1f-3
                # Every block must be a proper distribution.
                p = exp.(got); off = 0
                for K in dims
                    @test all(abs.(sum(p[off+1:off+K, :]; dims = 1) .- 1f0) .< 1f-4)
                    off += K
                end
            end
        end
    end

    # A table with enough categorical columns, trained at a wide batch: the
    # end-to-end shape of the NaN above.
    @testset "diffusion trains on many categorical columns" begin
        n = 2_000
        rng = MersenneTwister(7)
        cols = Dict{Symbol,Any}()
        for j in 1:3; cols[Symbol("num", j)] = randn(rng, n); end
        for (j, K) in enumerate([9, 16, 7, 15, 6, 5, 2])
            cols[Symbol("cat", j)] = rand(rng, string.(1:K), n)
        end
        tbl = NamedTuple(cols)
        model = fit(DiffusionGenerator(epochs = 2, batch_size = 1024,
                                       d_layers = [64, 64], num_timesteps = 50),
                    tbl; rng = MersenneTwister(1))
        @test model isa FittedDiffusionModel
        syn = sample(model, 200; rng = MersenneTwister(2))
        @test length(syn.num1) == 200
    end

    # ════════════════════════════════════════════════════════════════════════
    #  P H A S E   4  —  E V A L U A T I O N   S U I T E
    # ════════════════════════════════════════════════════════════════════════

    @testset "Evaluate" begin
        # Shared data for evaluation tests
        rng_eval = MersenneTwister(999)
        n = 200
        real_tbl = (;
            x   = randn(rng_eval, n),
            y   = randn(rng_eval, n) .* 2 .+ 1,
            cat = rand(rng_eval, ["a", "b", "c"], n),
        )
        model_eval = fit(CopulaGenerator(), real_tbl;
                         rng = MersenneTwister(1))
        synth_tbl = sample(model_eval, n; rng = MersenneTwister(2))

        # ── compare ─────────────────────────────────────────────────────
        @testset "compare" begin
            pb_eval = PrivacyBudget(epsilon = 1.0)
            cdf = DataFrame(a = randn(rng_eval, 300),
                            b = randn(rng_eval, 300) .* 2,
                            c = rand(rng_eval, ["x", "y", "z"], 300))

            @testset "one row per generator x metric" begin
                res = compare([CopulaGenerator(), CopulaGenerator(:gaussian)], cdf;
                              metrics = (fid = fidelity_score,
                                         dcr = (r, s) -> privacy_dcr(r, s).median),
                              n_seeds = 3, rng = MersenneTwister(1))
                @test length(res) == 4
                @test DataMimic.Tables.istable(res)
                @test all(r -> r.ok, res)
                @test all(r -> r.n_seeds == 3, res)
                @test all(r -> isfinite(r.mean) && isfinite(r.sd), res)
                @test all(r -> isfinite(r.fit_secs), res)
            end

            # Variants of one engine must be distinguishable, or the output is
            # unreadable for the most common use of the function.
            @testset "labels distinguish generator variants" begin
                res = compare([CopulaGenerator(), CopulaGenerator(:gaussian)], cdf;
                              metrics = (fid = fidelity_score,), n_seeds = 3,
                              rng = MersenneTwister(2))
                labels = unique(r.generator for r in res)
                @test length(labels) == 2
                @test any(l -> occursin("beta", l), labels)
                @test any(l -> occursin("gaussian", l), labels)

                # Explicit labels win.
                res2 = compare(["A" => CopulaGenerator(), "B" => CopulaGenerator()], cdf;
                               metrics = (fid = fidelity_score,), n_seeds = 3,
                               rng = MersenneTwister(3))
                @test Set(r.generator for r in res2) == Set(["A", "B"])

                # One engine at two budgets is a comparison the previous API
                # could not express at all: the budget was a single keyword
                # shared by every generator in the call.
                res_eps = compare([MSTGenerator(ε = 0.5), MSTGenerator(ε = 4.0)], cdf;
                                  metrics = (fid = fidelity_score,), n_seeds = 3,
                                  rng = MersenneTwister(5))
                eps_labels = unique(r.generator for r in res_eps)
                @test length(eps_labels) == 2
                @test any(l -> occursin("0.5", l), eps_labels)
                @test any(l -> occursin("4.0", l), eps_labels)

                # Genuinely identical generators are numbered rather than merged.
                res3 = compare([DPCopulaGenerator(privacy = pb_eval),
                                DPCopulaGenerator(privacy = pb_eval)], cdf;
                               metrics = (fid = fidelity_score,), n_seeds = 3,
                               rng = MersenneTwister(4))
                @test length(unique(r.generator for r in res3)) == 2
            end

            # A single failing engine must not destroy the whole comparison.
            @testset "a failing generator is isolated" begin
                res = compare([CopulaGenerator(), BoomGenerator()], cdf;
                              metrics = (fid = fidelity_score,), n_seeds = 2,
                              rng = MersenneTwister(5))
                good = only(filter(r -> !occursin("Boom", r.generator), res))
                bad  = only(filter(r ->  occursin("Boom", r.generator), res))
                @test good.ok && isfinite(good.mean)
                @test !bad.ok
                @test bad.n_failed == 2
                @test isnan(bad.mean)
                @test occursin("exploded", bad.error)
            end

            @testset "mixed public and private generators in one call" begin
                res = compare([CopulaGenerator(),
                               MSTGenerator(privacy = pb_eval)], cdf;
                              metrics = (fid = fidelity_score,), n_seeds = 3,
                              rng = MersenneTwister(6))
                @test length(res) == 2
                @test all(r -> r.ok, res)
            end

            @testset "metric may return a plain number" begin
                res = compare([CopulaGenerator()], cdf;
                              metrics = (agg = (r, s) -> fidelity_score(r, s).aggregate,),
                              n_seeds = 3, rng = MersenneTwister(7))
                @test only(res).ok
                @test isfinite(only(res).mean)
            end

            @testset "curried metric needs no anonymous function" begin
                # utility_tstr takes a target column, so it does not fit
                # compare's f(real, synth) shape on its own. The partially
                # applied form supplies it without a wrapper lambda.
                f = utility_tstr(:c)
                @test f isa Function

                res = compare([CopulaGenerator()], cdf;
                              metrics = (utility = f,),
                              n_seeds = 2, rng = MersenneTwister(11))
                @test only(res).ok
                @test isfinite(only(res).mean)

                # Currying must produce exactly what the direct call
                # produces. Compare them directly rather than through
                # `compare`: utility_tstr splits train/test at random, and
                # `compare` does not thread its rng into metric functions, so
                # routing this through `compare` would be testing that instead.
                #
                # Note also that the closure captures ONE rng and advances it
                # on every call, so a curried metric with a fixed rng does not
                # give every seed the same split. That is why each side here
                # builds its own generator and is called once.
                syn1 = sample(fit(CopulaGenerator(), cdf; rng = MersenneTwister(21)),
                              nrow(cdf); rng = MersenneTwister(22))
                direct  = utility_tstr(cdf, syn1, :c; rng = MersenneTwister(99))
                curried = utility_tstr(:c; rng = MersenneTwister(99))(cdf, syn1)
                @test direct.ratio ≈ curried.ratio
                @test direct.synth_score ≈ curried.synth_score

                # Keywords are forwarded to the underlying call.
                g = utility_tstr(:c; nrounds = 20)
                @test g(cdf, cdf) isa NamedTuple
                @test hasproperty(g(cdf, cdf), :ratio)
            end


            @testset "invalid input" begin
                @test_throws ArgumentError compare([], cdf)
                @test_throws ArgumentError compare([CopulaGenerator()], cdf;
                                                   metrics = NamedTuple())
                @test_throws ArgumentError compare([CopulaGenerator()], cdf; n_seeds = 0)
                @test_throws ArgumentError compare([CopulaGenerator()], "not a table")
            end
        end

        # ── fidelity_score ──────────────────────────────────────────────
        @testset "fidelity_score" begin
            @testset "basic output" begin
                fs = fidelity_score(real_tbl, synth_tbl)
                @test fs isa NamedTuple
                @test haskey(fs, :column_scores)
                @test haskey(fs, :column_metrics)
                @test haskey(fs, :correlation_score)
                @test haskey(fs, :aggregate)

                # All scores in [0, 1]
                for (_, v) in fs.column_scores
                    @test 0.0 ≤ v ≤ 1.0
                end
                @test 0.0 ≤ fs.correlation_score ≤ 1.0
                @test 0.0 ≤ fs.aggregate ≤ 1.0
            end

            @testset "KS for continuous, TVD for categorical" begin
                fs = fidelity_score(real_tbl, synth_tbl)
                @test fs.column_metrics[:x] == :ks
                @test fs.column_metrics[:y] == :ks
                @test fs.column_metrics[:cat] == :tvd
            end

            @testset "identical data → near-zero scores" begin
                fs = fidelity_score(real_tbl, real_tbl)
                @test fs.aggregate < 0.01
                @test fs.column_scores[:x] ≈ 0.0 atol = 1e-10
                @test fs.column_scores[:cat] ≈ 0.0 atol = 1e-10
                @test fs.correlation_score ≈ 0.0 atol = 1e-10
            end

            # Fitted models hold the whole learned artifact; the default struct
    # display dumps all of it (a small copula model printed >4000 chars, and a
    # diffusion model holds millions of weights). Display must stay a summary.
    @testset "fitted models display as a summary" begin
        dfs = DataFrame(a = randn(200), b = rand(["x", "y", "z"], 200),
                        id = ["r$i" for i in 1:200])
        m = fit(CopulaGenerator(), dfs; identifiers = [:id],
                fill = Dict(:id => :sequential), rng = MersenneTwister(3))

        long  = sprint(show, MIME"text/plain"(), m)
        short = sprint(show, m)

        @test length(long)  < 400
        @test length(short) < 200
        @test occursin("FittedCopulaModel", short)
        @test occursin("200 rows", long)
        @test occursin("identifier", long)
        # The learned artifact itself must not be dumped.
        @test !occursin("EmpiricalMarginal", long)
    end

    @testset "no shared columns → error" begin
                other = (; z = randn(50))
                @test_throws ArgumentError fidelity_score(real_tbl, other)
            end

            # REQ-EVL-003: a zero-variance column has no rank spread, so every
            # Spearman correlation involving it is 0/0.  Left unhandled, one
            # such column turns the correlation matrix — and through it the
            # headline aggregate — into NaN, even though each per-column score
            # computed fine.  Constant columns are common in real tables, so a
            # silent NaN here is a wrong answer rather than an edge case.
            @testset "constant column does not poison the score" begin
                n = 200
                rng_c = MersenneTwister(11)
                r = (; a = randn(rng_c, n), b = randn(rng_c, n), c = fill(1.0, n))
                s = (; a = randn(rng_c, n), b = randn(rng_c, n), c = fill(1.0, n))

                fs = fidelity_score(r, s)
                @test !isnan(fs.aggregate)
                @test !isnan(fs.correlation_score)

                # The constant column is still scored on its own …
                @test haskey(fs.column_scores, :c)
                @test fs.column_scores[:c] ≈ 0.0 atol = 1e-10
                # … and only dropped from the correlation term.
                @test :c in fs.correlation_excluded
                @test Set(fs.correlation_columns) == Set([:a, :b])

                # An all-constant table leaves nothing to correlate, which must
                # degrade gracefully to the 1-D mean rather than to NaN.
                r2 = (; a = fill(2.0, n), b = fill(5.0, n))
                fs2 = fidelity_score(r2, r2)
                @test !isnan(fs2.aggregate)
                @test fs2.correlation_score == 0.0
                @test isempty(fs2.correlation_columns)

                # A column constant in only one of the two tables also counts
                # as degenerate — the ranks are undefined on that side.
                r3 = (; a = randn(rng_c, n), b = randn(rng_c, n))
                s3 = (; a = randn(rng_c, n), b = fill(3.0, n))
                fs3 = fidelity_score(r3, s3)
                @test !isnan(fs3.aggregate)
                @test :b in fs3.correlation_excluded
            end

            @testset "non-table → error" begin
                @test_throws ArgumentError fidelity_score("bad", synth_tbl)
                @test_throws ArgumentError fidelity_score(real_tbl, "bad")
            end
        end

        # ── privacy_dcr ─────────────────────────────────────────────────
        @testset "privacy_dcr" begin
            @testset "basic output" begin
                dcr = privacy_dcr(real_tbl, synth_tbl)
                @test dcr isa NamedTuple
                @test haskey(dcr, :dcr)
                @test haskey(dcr, :median)
                @test haskey(dcr, :p5)
                @test haskey(dcr, :exact_matches)
                @test length(dcr.dcr) == n
                @test all(d -> d ≥ 0.0, dcr.dcr)
                @test dcr.median ≥ 0.0
                @test dcr.p5 ≥ 0.0
                @test dcr.exact_matches ≥ 0
            end

            @testset "identical data → all DCR = 0" begin
                dcr = privacy_dcr(real_tbl, real_tbl)
                @test all(==(0.0), dcr.dcr)
                @test dcr.exact_matches == n
            end

            @testset "no shared columns → error" begin
                other = (; z = randn(50))
                @test_throws ArgumentError privacy_dcr(real_tbl, other)
            end
        end

        # ── utility_tstr ────────────────────────────────────────────────
        @testset "utility_tstr" begin
            @testset "classification" begin
                tstr = utility_tstr(real_tbl, synth_tbl, :cat;
                                    nrounds = 20)
                @test tstr isa NamedTuple
                @test tstr.task == :classification
                @test 0.0 ≤ tstr.synth_score ≤ 1.0
                @test 0.0 ≤ tstr.real_score ≤ 1.0
                @test tstr.ratio ≥ 0.0
            end

            @testset "regression" begin
                tstr = utility_tstr(real_tbl, synth_tbl, :x;
                                    nrounds = 20)
                @test tstr.task == :regression
                @test tstr.synth_score ≥ 0.0  # RMSE
                @test tstr.real_score ≥ 0.0
                @test tstr.ratio > 0.0
            end

            @testset "missing target column → error" begin
                @test_throws ArgumentError utility_tstr(
                    real_tbl, synth_tbl, :nonexistent)
            end

            @testset "no shared features → error" begin
                only_target = (; cat = rand(["a", "b"], 50))
                @test_throws ArgumentError utility_tstr(
                    real_tbl, only_target, :cat)
            end
        end

        # ── jensen_shannon ─────────────────────────────────────────────
        @testset "jensen_shannon" begin
            @testset "basic output structure" begin
                js = jensen_shannon(real_tbl, synth_tbl)
                @test js isa NamedTuple
                @test haskey(js, :column_scores)
                @test haskey(js, :column_kinds)
                @test haskey(js, :mean)
                @test haskey(js, :aggregate)
                @test js.mean ≥ 0
                @test js.mean ≤ log(2) + 1e-10
                @test js.aggregate == js.mean
                @test length(js.column_scores) > 0
            end

            @testset "identical data → JSD ≈ 0" begin
                js = jensen_shannon(real_tbl, real_tbl)
                @test js.mean < 0.01
                for (_, v) in js.column_scores
                    @test v < 0.01
                end
            end

            @testset "numeric vs categorical detection" begin
                js = jensen_shannon(real_tbl, synth_tbl)
                @test js.column_kinds[:x] == :numeric
                @test js.column_kinds[:cat] == :categorical
            end

            @testset "errors" begin
                @test_throws ArgumentError jensen_shannon("bad", synth_tbl)
                @test_throws ArgumentError jensen_shannon(real_tbl, "bad")
                other = (; zzz = rand(50))
                @test_throws ArgumentError jensen_shannon(real_tbl, other)
                @test_throws ArgumentError jensen_shannon(real_tbl, synth_tbl; n_bins = 0)
            end
        end

        # ── pairwise_marginal_error ────────────────────────────────────
        @testset "pairwise_marginal_error" begin
            @testset "basic output structure" begin
                pme = pairwise_marginal_error(real_tbl, synth_tbl)
                @test pme isa NamedTuple
                @test haskey(pme, :pair_scores)
                @test haskey(pme, :mean)
                @test haskey(pme, :worst_pair)
                @test haskey(pme, :worst_score)
                @test haskey(pme, :n_pairs)
                @test pme.mean ≥ 0
                @test pme.mean ≤ 1.0
                @test pme.n_pairs > 0
                @test pme.worst_score ≥ pme.mean
            end

            @testset "identical data → low error" begin
                pme = pairwise_marginal_error(real_tbl, real_tbl)
                @test pme.mean < 0.05
            end

            @testset "order=3" begin
                # Need at least 3 shared columns
                pme3 = pairwise_marginal_error(real_tbl, synth_tbl; order = 3)
                @test pme3.n_pairs > 0
                @test length(first(keys(pme3.pair_scores))) == 3
            end

            @testset "errors" begin
                @test_throws ArgumentError pairwise_marginal_error("bad", synth_tbl)
                @test_throws ArgumentError pairwise_marginal_error(
                    real_tbl, synth_tbl; order = 4)
                small = (; a = [1, 2])
                @test_throws ArgumentError pairwise_marginal_error(small, small)
            end
        end

        # ── privacy_utility_sweep ──────────────────────────────────────
        @testset "privacy_utility_sweep" begin
            @testset "basic sweep" begin
                sweep_data = (; a = randn(200), b = rand(["x","y"], 200))
                results = privacy_utility_sweep(
                    DPCopulaGenerator, sweep_data, [1.0, 10.0],
                    fidelity_score;
                    rng = MersenneTwister(42)
                )
                @test length(results) == 2
                @test results[1].epsilon == 1.0
                @test results[2].epsilon == 10.0
                @test results[1].delta == 1e-5
                # Higher ε should generally give better fidelity (lower score)
                @test results[1].metric_result isa NamedTuple
                @test results[2].metric_result isa NamedTuple
            end

            @testset "works with jensen_shannon" begin
                sweep_data = (; a = randn(200), b = rand(["x","y"], 200))
                results = privacy_utility_sweep(
                    DPCopulaGenerator, sweep_data, [5.0],
                    jensen_shannon;
                    rng = MersenneTwister(42)
                )
                @test length(results) == 1
                @test haskey(results[1].metric_result, :mean)
            end

            @testset "errors" begin
                sweep_data = (; a = randn(100))
                @test_throws ArgumentError privacy_utility_sweep(
                    DPCopulaGenerator, sweep_data, Float64[],
                    fidelity_score)
                @test_throws ArgumentError privacy_utility_sweep(
                    DPCopulaGenerator, sweep_data, [-1.0],
                    fidelity_score)

                # A constructed generator has already fixed its ε, so
                # sweeping it is meaningless rather than merely unsupported.
                @test_throws ArgumentError privacy_utility_sweep(
                    DPCopulaGenerator(ε = 1.0), sweep_data, [1.0],
                    fidelity_score)

                # And a constructor returning a public generator would give a
                # flat curve that looks like a finding.
                @test_throws ArgumentError privacy_utility_sweep(
                    b -> CopulaGenerator(), sweep_data, [1.0], fidelity_score)
            end
        end
    end

end

# Aqua + JET, backing the quality badges in the README.
include("quality.jl")
