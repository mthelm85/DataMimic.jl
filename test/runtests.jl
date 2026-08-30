using DataMimic
using DataFrames
using Test
using Random
using LinearAlgebra: eigvals
using Lux, Zygote

@testset "DataMimic.jl v2.0" begin

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
            @test MSTGenerator().max_marginal_order == 2
            @test MSTGenerator(3).max_marginal_order == 3
            @test_throws ArgumentError MSTGenerator(5)
        end

        @testset "DiffusionGenerator" begin
            dg = DiffusionGenerator()
            @test dg.epochs == 100
            @test dg.batch_size == 512
            @test dg.dp == false
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
        @test !(:encoded in model.copula_columns)

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
    # AutoGenerator — non-private
    # ════════════════════════════════════════════════════════════════════════
    @testset "AutoGenerator non-private" begin
        df = make_df()
        model = fit(AutoGenerator(), df)
        @test model isa FittedCopulaModel
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
    @testset "privacy validation" begin
        df = make_df()
        pb = PrivacyBudget(epsilon = 1.0)

        # Public generator rejects privacy budget
        @test_throws ArgumentError fit(CopulaGenerator(), df; privacy = pb)

        # Private generator requires privacy budget
        @test_throws ArgumentError fit(MSTGenerator(), df)
        @test_throws ArgumentError fit(DPCopulaGenerator(), df)
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
    @testset "single numeric column (no copula)" begin
        df = DataFrame(x = randn(100), cat = rand(["a", "b", "c"], 100))
        model = @test_logs (:warn,) fit(CopulaGenerator(), df)
        @test isnothing(model.copula)
        syn = sample(model, 50)
        @test nrow(syn) == 50
    end

    @testset "all categorical (no copula)" begin
        df = DataFrame(a = rand(["x", "y", "z"], 100),
                        b = rand(["p", "q"], 100))
        model = @test_logs (:warn,) fit(CopulaGenerator(), df)
        @test isnothing(model.copula)
        syn = sample(model, 30)
        @test nrow(syn) == 30
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
            model = fit(MSTGenerator(), tbl;
                        privacy = pb, rng = MersenneTwister(42))
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
            model = fit(MSTGenerator(), df;
                        privacy = pb, rng = MersenneTwister(42))
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
            model = fit(MSTGenerator(), df;
                        privacy = pb, identifiers = [:id],
                        fill = Dict(:id => :sequential),
                        rng = MersenneTwister(42))
            syn = sample(model, 30)
            @test :id in Symbol.(names(syn))
            @test syn.id[1] == "id_1"
            @test !(:id in model.stat_columns)
        end

        @testset "3-way marginals warn and fallback" begin
            model = @test_logs (:warn, r"3-way") fit(
                MSTGenerator(3), tbl;
                privacy = pb, rng = MersenneTwister(42))
            @test model isa FittedMSTModel
        end

        @testset "reproducibility" begin
            m1 = fit(MSTGenerator(), tbl; privacy = pb, rng = MersenneTwister(1))
            m2 = fit(MSTGenerator(), tbl; privacy = pb, rng = MersenneTwister(1))

            s1 = sample(m1, 40; rng = MersenneTwister(99))
            s2 = sample(m2, 40; rng = MersenneTwister(99))
            @test s1.gender == s2.gender
            @test s1.age    == s2.age
        end

        @testset "single column" begin
            single = (; x = rand(rng_data, ["a", "b", "c"], 50))
            model = fit(MSTGenerator(), single;
                        privacy = pb, rng = MersenneTwister(42))
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
            model = fit(MSTGenerator(), cat_tbl;
                        privacy = pb, rng = MersenneTwister(42))
            syn = sample(model, 40)
            @test all(v -> v ∈ ["x", "y", "z"], syn.a)
            @test all(v -> v ∈ ["p", "q"], syn.b)
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
            model = fit(DPCopulaGenerator(), tbl;
                        privacy = pb, rng = MersenneTwister(42))
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
            model = fit(DPCopulaGenerator(), df;
                        privacy = pb, rng = MersenneTwister(42))
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
                DPCopulaGenerator(), single;
                privacy = pb, rng = MersenneTwister(42))
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
                DPCopulaGenerator(), cat_tbl;
                privacy = pb, rng = MersenneTwister(42))
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
            m1 = fit(DPCopulaGenerator(), tbl; privacy = pb,
                     rng = MersenneTwister(1))
            m2 = fit(DPCopulaGenerator(), tbl; privacy = pb,
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
            model = fit(DPCopulaGenerator(), df;
                        privacy = pb, identifiers = [:id],
                        fill = Dict(:id => :sequential_int),
                        rng = MersenneTwister(42))
            syn = sample(model, 25)
            @test syn.id == collect(1:25)
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # AutoGenerator — private dispatch
    # ════════════════════════════════════════════════════════════════════════
    @testset "AutoGenerator private dispatch" begin
        pb = PrivacyBudget(epsilon = 2.0, delta = 1e-5)

        @testset "small N, categorical-heavy → MSTGenerator" begin
            # n=50, 3 categorical + 1 numeric: cat_frac = 3/4 > 0.5
            df = DataFrame(
                a = rand(["x","y","z"], 50),
                b = rand(["p","q"], 50),
                c = rand([true, false], 50),
                d = randn(50),
            )
            model = fit(AutoGenerator(), df;
                        privacy = pb, rng = MersenneTwister(42))
            @test model isa FittedMSTModel
        end

        @testset "small N, continuous-heavy → DPCopulaGenerator" begin
            # n=50, 3 numeric + 1 categorical: cat_frac = 1/4 ≤ 0.5
            df = DataFrame(
                a = randn(50),
                b = randn(50),
                c = randn(50),
                d = rand(["x", "y"], 50),
            )
            model = fit(AutoGenerator(), df;
                        privacy = pb, rng = MersenneTwister(42))
            @test model isa FittedDPCopulaModel
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
            model = fit(MSTGenerator(), tbl; privacy = pb,
                        rng = MersenneTwister(42))
            @test isapprox(model.missingness[:x], 0.1, atol = 0.01)
            syn = sample(model, 500)
            p_miss = count(ismissing, syn.x) / 500
            @test p_miss > 0.0
            @test Missing <: eltype(syn.x)
        end

        @testset "DPCopula" begin
            model = fit(DPCopulaGenerator(), tbl; privacy = pb,
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
            model = fit(MSTGenerator(), tbl; privacy = pb,
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
            model = fit(DPCopulaGenerator(), tbl; privacy = pb,
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

        # Private generators reject missing privacy
        @test_throws ArgumentError fit(MSTGenerator(), tbl)
        @test_throws ArgumentError fit(DPCopulaGenerator(), tbl)

        # sample with n < 1
        model = fit(MSTGenerator(), tbl; privacy = pb,
                    rng = MersenneTwister(42))
        @test_throws ArgumentError sample(model, 0)

        model2 = fit(DPCopulaGenerator(), tbl; privacy = pb,
                     rng = MersenneTwister(42))
        @test_throws ArgumentError sample(model2, 0)

        # DiffusionGenerator privacy validation
        @test_throws ArgumentError fit(DiffusionGenerator(; dp = true),
                                       tbl)   # dp=true needs a budget
        @test_throws ArgumentError fit(DiffusionGenerator(; dp = false),
                                       tbl; privacy = pb)   # dp=false rejects budget
    end

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2 synthesize convenience
    # ════════════════════════════════════════════════════════════════════════
    @testset "synthesize with private generators" begin
        pb = PrivacyBudget(epsilon = 2.0)
        df = DataFrame(x = randn(80), y = rand(["a", "b", "c"], 80))

        syn = synthesize(MSTGenerator(), df, 40;
                         privacy = pb, rng = MersenneTwister(42))
        @test syn isa DataFrame
        @test nrow(syn) == 40

        syn2 = synthesize(DPCopulaGenerator(), df, 40;
                          privacy = pb, rng = MersenneTwister(42))
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

        @testset "fit + sample with dp=true" begin
            model = fit(DiffusionGenerator(; dp = true, epochs = 2,
                                             batch_size = 16),
                        tbl; privacy = pb, rng = MersenneTwister(42))
            @test model isa FittedDiffusionModel
            syn = sample(model, 20)
            @test length(syn.x) == 20
            @test all(c -> c ∈ ["a", "b", "c"], syn.cat)
        end

        @testset "dp=true without budget → error" begin
            @test_throws ArgumentError fit(
                DiffusionGenerator(; dp = true, epochs = 2),
                tbl; rng = MersenneTwister(42))
        end

        @testset "dp=false with budget → error" begin
            @test_throws ArgumentError fit(
                DiffusionGenerator(; dp = false, epochs = 2),
                tbl; privacy = pb, rng = MersenneTwister(42))
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
            model = fit(DiffusionGenerator(; dp = true, epochs = 2,
                                             batch_size = 1, num_timesteps = 10,
                                             hidden_dim = 8, n_blocks = 1),
                        small; privacy = pb, rng = MersenneTwister(42))
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
    # AutoGenerator dispatch to DiffusionGenerator
    # ════════════════════════════════════════════════════════════════════════
    @testset "AutoGenerator → DiffusionGenerator" begin
        @testset "non-private: D > 30 → Diffusion" begin
            # Build a table with 35 numeric columns
            cols_nt = NamedTuple{Tuple(Symbol.("c" .* string.(1:35)))}(
                Tuple(randn(Float32, 60) for _ in 1:35))
            model = fit(AutoGenerator(), cols_nt; rng = MersenneTwister(42))
            @test model isa FittedDiffusionModel
        end
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
                    DPCopulaGenerator(), sweep_data, [1.0, 10.0],
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
                    DPCopulaGenerator(), sweep_data, [5.0],
                    jensen_shannon;
                    rng = MersenneTwister(42)
                )
                @test length(results) == 1
                @test haskey(results[1].metric_result, :mean)
            end

            @testset "errors" begin
                sweep_data = (; a = randn(100))
                @test_throws ArgumentError privacy_utility_sweep(
                    DPCopulaGenerator(), sweep_data, Float64[],
                    fidelity_score)
                @test_throws ArgumentError privacy_utility_sweep(
                    DPCopulaGenerator(), sweep_data, [-1.0],
                    fidelity_score)
            end
        end
    end

end
