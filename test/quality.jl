# ─── Package quality checks: Aqua + JET ────────────────────────────────────
#
# These back the badges in the README, so they run as part of the normal test
# suite rather than as a separate opt-in job.

using Aqua
using JET

const Ext = Base.get_extension(DataMimic, :DataMimicLuxExt)

@testset "Quality" begin

    # ── Aqua ────────────────────────────────────────────────────────────────
    # Method ambiguities, unbound type parameters, undefined exports, stale
    # and under-constrained dependencies, type piracy, and tasks left running
    # by precompilation. The whole suite passes, so none of it is disabled.
    @testset "Aqua" begin
        Aqua.test_all(DataMimic)
    end

    # ── JET ─────────────────────────────────────────────────────────────────
    # Deliberately targeted rather than `JET.test_package`. Whole-package
    # analysis enters EvoTrees, Missings and Base.Broadcast through our
    # abstractly-typed public entry points, and nearly every report it
    # produces belongs to those packages rather than to this one. The two
    # reports that did land in our own code were false positives — a
    # `Float64(::String)` in the regression branch of `utility_tstr`, which a
    # runtime branch makes unreachable for string targets, and a tuple-length
    # BoundsError in `_combinations`, which builds its result vector from the
    # same `k` it converts with. Asserting zero reports package-wide would
    # therefore mean either a failing test or a pile of exclusions that hide
    # real regressions later.
    #
    # Instead: call the internals with concrete argument types, where a report
    # means something. `@test_call` catches would-be method errors, and
    # `@test_opt` catches dynamic dispatch — which is what made Zygote emit
    # enormous amounts of code and dominated diffusion training time.
    @testset "JET" begin
        rng = MersenneTwister(0)
        n   = 200
        xf  = randn(n)
        xi  = rand(1:50, n)
        xs  = rand(["a", "b", "c"], n)
        xb  = rand(Bool, n)

        @testset "core" begin
            @test_call DataMimic._detect_column_type(xf, Float64)
            @test_call DataMimic._detect_column_type(xi, Int)
            @test_call DataMimic._detect_column_type(xs, String)
            @test_call DataMimic._detect_column_type(xb, Bool)
            @test_call DataMimic._symmetric_gaussian_noise(5, 1.0, rng)
            @test_opt  DataMimic._symmetric_gaussian_noise(5, 1.0, rng)
        end

        @testset "evaluate" begin
            E = DataMimic.Evaluate
            @test_call E._combinations([:a, :b, :c], 2)
            @test_call E._discretize_column(xf, 10; ref_col = xf)
            @test_call E._discretize_column(xs, 10; ref_col = xs)
            @test_opt  E._discretize_column(xf, 10; ref_col = xf)
        end

        # The diffusion hot path, which runs once per training step per batch
        # and is where dynamic dispatch is most expensive.
        @testset "diffusion hot path" begin
            cat_dims = [4, 3, 2]
            d_cat    = sum(cat_dims)
            bs       = 8
            dev      = Lux.cpu_device()

            betas, _ = Ext._cosine_schedule(100)
            sched    = Ext._schedule_constants(betas)
            plan     = Ext._block_plan(cat_dims, dev)
            log_K    = Ext._log_K_vector(cat_dims)
            coef     = Ext._batch_coefs(Ext._device_schedule(sched, dev),
                                        rand(rng, 1:100, bs))

            x_oh = zeros(Float32, d_cat, bs)
            let off = 0
                for K in cat_dims
                    for b in 1:bs
                        x_oh[off + rand(rng, 1:K), b] = 1f0
                    end
                    off += K
                end
            end
            log_x0   = Ext._to_log_onehot(x_oh)
            unnormed = randn(rng, Float32, d_cat, bs)

            @test_call Ext._cosine_schedule(100)
            @test_call Ext._block_plan(cat_dims, dev)
            @test_call Ext._log_1_min_a(-0.5)
            @test_call Ext._log_add_exp(0f0, -1f0)

            # Written out rather than looped over closures: wrapping these in
            # a closure would attribute any report to the closure and risks
            # analysing that instead of the function under test.
            @test_call Ext._block_log_normalize(unnormed, plan)
            @test_opt  Ext._block_log_normalize(unnormed, plan)

            @test_call Ext._q_posterior(log_x0, log_x0, plan, coef, log_K)
            @test_opt  Ext._q_posterior(log_x0, log_x0, plan, coef, log_K)

            @test_call Ext._p_pred(unnormed, log_x0, plan, coef, log_K)
            @test_opt  Ext._p_pred(unnormed, log_x0, plan, coef, log_K)

            @test_call Ext._multinomial_kl(log_x0, log_x0)
            @test_opt  Ext._multinomial_kl(log_x0, log_x0)
        end
    end
end
