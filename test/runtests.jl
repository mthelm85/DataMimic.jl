using DataMimic
using DataFrames
using Test

@testset "DataMimic.jl" begin

    # ── Helpers ────────────────────────────────────────────────────────────────
    function make_df(n=200)
        DataFrame(
            x_float  = randn(n),
            x_int    = rand(1:50, n),
            x_cat    = rand(["a", "b", "c"], n),
            x_bool   = rand([true, false], n),
            x_const  = fill("same", n),
        )
    end

    # ── Column type detection ──────────────────────────────────────────────────
    @testset "detect_column_type" begin
        using DataMimic: detect_column_type

        @test detect_column_type(fill(1.0, 5))           == :constant
        @test detect_column_type([1.0, 2.0])              == :binary
        # Float64 with non-whole values → continuous
        @test detect_column_type([1.1, 2.2, 3.3])         == :continuous
        @test detect_column_type([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]) == :continuous
        # Float64 where all values are whole numbers → integer (per spec)
        @test detect_column_type([1.0, 2.0, 3.0])         == :integer
        @test detect_column_type([1.0, 2.0, 3.0, 4.0])   == :integer
        @test detect_column_type([1.0, 2.0, 3.0, 4.0, 5.0]) == :integer
        # True Int eltype → integer
        @test detect_column_type([1, 2, 3])               == :integer
        @test detect_column_type(Int[1, 2, 3, 4])         == :integer
        # Categorical / binary
        @test detect_column_type(["a", "b", "c"])         == :categorical
        @test detect_column_type(["a", "b"])               == :binary
        @test detect_column_type([true, false, true])      == :binary  # exactly 2 unique
        # Union{T,Missing}: 2 non-missing unique values → binary
        col_binary_miss = Union{Float64,Missing}[1.0, missing, 3.0]
        @test detect_column_type(col_binary_miss)          == :binary
        # Union{T,Missing}: 3+ non-missing, non-whole values → continuous
        col_cont_miss = Union{Float64,Missing}[1.1, missing, 3.3, 4.4]
        @test detect_column_type(col_cont_miss)            == :continuous
        # Entirely missing
        all_miss = Union{Float64,Missing}[missing, missing]
        @test detect_column_type(all_miss)                 == :constant
    end

    # ── fit / SynthModel ───────────────────────────────────────────────────────
    @testset "fit" begin
        df = make_df()
        model = fit(df)

        @test model isa SynthModel
        @test model.column_names == [:x_float, :x_int, :x_cat, :x_bool, :x_const]
        @test length(model.column_types) == 5
        @test model.column_types[1] == :continuous
        @test model.column_types[2] == :integer
        @test model.column_types[3] == :categorical
        @test model.column_types[4] == :binary
        @test model.column_types[5] == :constant
        @test model.nrows_original  == 200
        @test all(values(model.missingness) .== 0.0)
        @test !isnothing(model.copula)
        @test :x_float in model.copula_columns
        @test :x_int   in model.copula_columns
    end

    # ── Error conditions ───────────────────────────────────────────────────────
    @testset "fit errors" begin
        @test_throws ErrorException fit(DataFrame())
        @test_throws ErrorException fit(DataFrame(a=Int[]))
    end

    @testset "sample errors" begin
        model = fit(make_df())
        @test_throws ErrorException sample(model, 0)
        @test_throws ErrorException sample(model, -1)
    end

    # ── sample output shape and types ──────────────────────────────────────────
    @testset "sample output" begin
        df    = make_df(100)
        model = fit(df)
        syn   = sample(model, 150)

        @test syn isa DataFrame
        @test nrow(syn) == 150
        @test ncol(syn) == 5
        @test names(syn) == names(df)

        # Float column is still Float64
        @test eltype(syn.x_float) == Float64

        # Integer column has no fractional values
        @test all(v -> v == round(v), syn.x_int)

        # Categorical column only contains original levels
        original_levels = Set(df.x_cat)
        @test all(v -> v in original_levels, syn.x_cat)

        # Constant column is all "same"
        @test all(==(first(df.x_const)), syn.x_const)
    end

    # ── Missing value handling ─────────────────────────────────────────────────
    @testset "missingness" begin
        n  = 500
        col = Vector{Union{Float64, Missing}}(randn(n))
        col[1:50] .= missing     # 10 % missing rate
        df    = DataFrame(x=col, y=randn(n))
        model = fit(df)

        @test isapprox(model.missingness[:x], 0.1, atol=0.01)

        syn = sample(model, 1000)
        p_obs = count(ismissing, syn.x) / 1000
        @test isapprox(p_obs, 0.1, atol=0.05)
        @test Missing <: eltype(syn.x)
    end

    # ── synthesize convenience wrapper ─────────────────────────────────────────
    @testset "synthesize" begin
        df  = make_df(50)
        syn = synthesize(df, 60)
        @test syn isa DataFrame
        @test nrow(syn) == 60
        @test names(syn) == names(df)
    end

    # ── Single numeric column (no copula) ─────────────────────────────────────
    @testset "single numeric column" begin
        df = DataFrame(x=randn(100), cat=rand(["a","b","c"], 100))
        @test_logs (:warn,) fit(df)   # should warn about skipping copula
        model = fit(df)
        @test isnothing(model.copula)
        syn = sample(model, 50)
        @test nrow(syn) == 50
    end

    # ── All categorical (no copula fallback) ──────────────────────────────────
    @testset "all categorical" begin
        df = DataFrame(a=rand(["x","y","z"], 100), b=rand(["p","q"], 100))
        @test_logs (:warn,) fit(df)
        model = fit(df)
        @test isnothing(model.copula)
        syn = sample(model, 30)
        @test nrow(syn) == 30
    end

end
