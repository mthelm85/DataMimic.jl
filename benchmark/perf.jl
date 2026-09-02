# ─── DataMimic performance suite ───────────────────────────────────────────
#
# A regression harness for run time, alongside `run_benchmarks.jl`, which
# measures output *quality*. Both matter and they fail independently: an
# engine can get faster and worse, or slower and better.
#
#   julia --project=benchmark benchmark/perf.jl              # run + compare
#   julia --project=benchmark benchmark/perf.jl --save       # record baseline
#   julia --project=benchmark benchmark/perf.jl --cpu        # force CPU
#   julia --project=benchmark benchmark/perf.jl --slow       # include DP-SGD
#   julia --project=benchmark benchmark/perf.jl --filter=mst
#
# Exits non-zero when a case regresses past the threshold, so it can gate CI.
#
# ── Why the harness looks like this ────────────────────────────────────────
#
# Every guard below exists because its absence produced a confidently wrong
# number during the performance investigation this file came out of:
#
#   1. Warm up before timing. Julia compiles on first call and caches the
#      specialization for the rest of the process, so an unwarmed "first vs
#      later" comparison can report a *negative* per-epoch cost.
#   2. Synchronize BEFORE starting the clock, not just after. CUDA is
#      asynchronous, so a timing block that only syncs at the end also waits
#      for work queued by whatever ran before it. This inflated one function's
#      apparent share of run time to 21% when its real share was near zero.
#   3. Compare on the MINIMUM, reporting median and max alongside. Timing
#      noise is one-sided — it only ever adds time — so the minimum is the
#      most stable estimate of true cost. Comparing medians flagged a false
#      regression on an immediate re-run of an unchanged tree.
#   5. Warm every case before timing any of them. Per-case warmup does not
#      cover process-level lazy initialization, which lands on whichever case
#      happens to run first.
#   6. Repeat sub-millisecond cases until each sample clears the clock's
#      resolution. `time()` on Windows is coarse enough that a 1 ms case reads
#      as 1 ms or 2 ms, which is a 100% "regression" from rounding alone.
#   7. Report median drift across all cases. A machine under load slows
#      everything at once; without that summary it looks like ten independent
#      regressions, which is exactly what happened the first time this ran
#      after a real change.
#   4. Record the machine and device in the baseline, and refuse to compare
#      across different ones. CPU and GPU differ by up to 28x on the same
#      case in opposite directions depending on batch size.

using DataMimic
using DataFrames
using Printf
using Random
using Statistics
using Dates
using TOML
using Lux, Zygote

const ARGS_SET   = Set(ARGS)
const FORCE_CPU  = "--cpu"   in ARGS_SET
const SAVE_BASE  = "--save"  in ARGS_SET
const RUN_SLOW   = "--slow"  in ARGS_SET
const FILTER     = something(findfirst(a -> startswith(a, "--filter="), ARGS), 0) == 0 ?
                   "" : split(ARGS[findfirst(a -> startswith(a, "--filter="), ARGS)], "=")[2]

# Regression thresholds. Wide enough to survive ordinary machine noise;
# anything real in this codebase has been far larger than 25%.
const REGRESS_FACTOR   = 1.25
const IMPROVE_FACTOR   = 0.80
const BASELINE_PATH    = joinpath(@__DIR__, "perf_baseline.toml")

# Each timed sample must span at least this long, so clock resolution cannot
# dominate. Cheap cases are looped internally and the total divided out.
const MIN_SAMPLE_SECONDS = 0.05

# ── Device ─────────────────────────────────────────────────────────────────

const GPU_SYNC = Ref{Function}(() -> nothing)

if !FORCE_CPU
    try
        @eval using LuxCUDA
        if LuxCUDA.functional()
            GPU_SYNC[] = () -> LuxCUDA.CUDA.synchronize()
        end
    catch
        @info "LuxCUDA unavailable; running on CPU."
    end
end

sync() = GPU_SYNC[]()

const DEVICE_NAME = let
    d = Lux.gpu_device(; force = false)
    d isa Lux.CPUDevice ? "cpu" : "gpu"
end

# ── Harness ────────────────────────────────────────────────────────────────

struct Case
    name::String
    group::String
    f::Function      # the work to time; takes no arguments
    reps::Int
    slow::Bool
end
Case(name, group, f; reps = 7, slow = false) = Case(name, group, f, reps, slow)

"""
Time `case.f` and return (median, min, max) seconds.

Warms up first so compilation is excluded, and synchronizes the device on
both sides of the clock so a case cannot absorb work queued by its
predecessor.
"""
function measure(case::Case)
    case.f()                       # warm up: compile, allocate, populate caches
    sync()

    # Pick an inner repeat count so each sample clears the clock's resolution.
    t0 = time(); case.f(); sync()
    single = max(time() - t0, 1e-9)
    k = max(1, ceil(Int, MIN_SAMPLE_SECONDS / single))

    times = Float64[]
    for _ in 1:case.reps
        sync()                     # drain anything still pending BEFORE timing
        t0 = time()
        for _ in 1:k
            case.f()
        end
        sync()                     # and only then stop the clock
        push!(times, (time() - t0) / k)
    end
    return median(times), minimum(times), maximum(times)
end

# ── Fixtures ───────────────────────────────────────────────────────────────
#
# Fixed seeds throughout: a case that changes its own input between runs is
# measuring the input, not the code.

function mixed_table(n; seed = 42)
    rng = MersenneTwister(seed)
    DataFrame(
        num1 = randn(rng, n), num2 = randn(rng, n) .* 3 .+ 1,
        num3 = rand(rng, n) .* 100, num4 = randn(rng, n),
        cat1 = rand(rng, ["a", "b", "c"], n),
        cat2 = rand(rng, ["p", "q"], n),
        cat3 = rand(rng, string.(1:8), n),
        bin1 = rand(rng, [true, false], n),
    )
end

function wide_table(n; seed = 7)
    rng = MersenneTwister(seed)
    df = DataFrame()
    for j in 1:5;  df[!, Symbol("num", j)] = randn(rng, n); end
    for (j, K) in enumerate([9, 16, 16, 7, 15, 6, 5, 2, 42])
        df[!, Symbol("cat", j)] = rand(rng, string.(1:K), n)
    end
    df
end

"""
A table with genuinely high-cardinality categoricals, Zipf-distributed so the
tails are sparse rather than uniform.

Categorical columns get one bin per level - `MST_DEFAULT_BINS` caps only
numeric columns - and `_edge_scores` builds an `n_bins[i] x n_bins[j]` table
for every column pair, so MST cost is quadratic in level count. Nothing else
in this suite exercises that.
"""
function hicard_table(n; seed = 11)
    rng = MersenneTwister(seed)
    df = DataFrame(num1 = randn(rng, n), num2 = randn(rng, n))
    for (j, K) in enumerate([512, 256, 64])
        w = [1.0 / l for l in 1:K]
        w ./= sum(w)
        cum = cumsum(w)
        df[!, Symbol("cat", j)] =
            [string(searchsortedfirst(cum, rand(rng))) for _ in 1:n]
    end
    return df
end

const TBL_SMALL = mixed_table(2_000)
const TBL_MED   = mixed_table(20_000)
const TBL_WIDE  = wide_table(20_000)   # 9 categorical blocks
const TBL_HICARD = hicard_table(20_000)
const SYN_MED   = mixed_table(20_000; seed = 99)

const BUDGET = PrivacyBudget(epsilon = 1.0, delta = 1e-5)
const RNG()  = MersenneTwister(1234)

# ── Cases ──────────────────────────────────────────────────────────────────

function build_cases()
    cs = Case[]

    # Public engine — the default path, and the one most users hit.
    push!(cs, Case("copula_beta/fit", "engines",
        () -> fit(CopulaGenerator(), TBL_MED; rng = RNG())))
    push!(cs, Case("copula_gaussian/fit", "engines",
        () -> fit(CopulaGenerator(:gaussian), TBL_MED; rng = RNG())))
    let m = fit(CopulaGenerator(), TBL_MED; rng = RNG())
        push!(cs, Case("copula_beta/sample_20k", "engines",
            () -> sample(m, 20_000; rng = RNG())))
    end

    # Private engines.
    push!(cs, Case("mst/fit", "engines",
        () -> fit(MSTGenerator(), TBL_MED; privacy = BUDGET, rng = RNG()), reps = 3))
    let m = fit(MSTGenerator(), TBL_MED; privacy = BUDGET, rng = RNG())
        push!(cs, Case("mst/sample_20k", "engines",
            () -> sample(m, 20_000; rng = RNG())))
    end
    push!(cs, Case("dpcopula/fit", "engines",
        () -> fit(DPCopulaGenerator(), TBL_MED; privacy = BUDGET, rng = RNG()), reps = 3))

    # High-cardinality categoricals: MST's cost is quadratic in level count,
    # and no other case here has a column wider than 42 levels. Domain
    # compression runs before tree selection and shrinks the domains it scores
    # over, which is the only reason this case is cheap enough to keep in a
    # routine suite - measured at 5.81s per fit with compression off against
    # 0.18s with it on. A regression here is therefore as likely to mean
    # compression stopped firing as that something got slower.
    push!(cs, Case("mst/fit_hicard", "engines",
        () -> fit(MSTGenerator(), TBL_HICARD; privacy = BUDGET, rng = RNG()), reps = 3))
    let m = fit(MSTGenerator(), TBL_HICARD; privacy = BUDGET, rng = RNG())
        push!(cs, Case("mst/sample_hicard_20k", "engines",
            () -> sample(m, 20_000; rng = RNG()), reps = 3))
    end

    # Diffusion. Epoch counts are deliberately small — this tracks cost per
    # unit of work, not convergence.
    #
    # TBL_WIDE used to be excluded here: 9 independent uniformly-random
    # categoricals drove training to a NaN loss at epoch 1, and the comment
    # in its place called that an open bug. It is not open. The cause was
    # `_block_log_normalize` subtracting a single GLOBAL maximum from raw
    # logits, so a block far below that maximum underflowed to zero and every
    # downstream KL became NaN — which needs several blocks and a wide batch
    # to show, exactly this table. Fixed with a per-block maximum, and
    # regression-tested under "block log-softmax stability".
    #
    # Re-verified before enabling: 3 epochs at (batch 4096, lr 1e-3),
    # (512, 1e-3) and (4096, 1e-2) all train to finite decreasing losses and
    # sample finite output. The wide case is kept below as the suite's only
    # diffusion coverage of many categorical blocks.
    let g = DiffusionGenerator(; epochs = 3, batch_size = 4096,
                               d_layers = [256, 1024, 1024, 256], num_timesteps = 100)
        push!(cs, Case("diffusion/fit_3ep", "diffusion",
            () -> fit(g, TBL_MED; rng = RNG()), reps = 3))
    end
    let m = fit(DiffusionGenerator(; epochs = 2, batch_size = 4096,
                                   d_layers = [256, 256], num_timesteps = 100),
                TBL_MED; rng = RNG())
        push!(cs, Case("diffusion/sample_10k", "diffusion",
            () -> sample(m, 10_000; rng = RNG()), reps = 3))
    end

    # Many categorical blocks rather than a bigger network: a deliberately
    # small MLP, so what moves here is the per-block multinomial machinery.
    let g = DiffusionGenerator(; epochs = 3, batch_size = 4096,
                               d_layers = [256, 256], num_timesteps = 100)
        push!(cs, Case("diffusion/fit_wide_3ep", "diffusion",
            () -> fit(g, TBL_WIDE; rng = RNG()), reps = 3))
    end

    # DP-SGD used to be gated behind --slow, at ~300x the cost of standard
    # training per epoch. Ghost clipping brought it within a small multiple of
    # the ordinary path, so it runs by default — and this case is what would
    # catch a regression back to per-example gradients.
    let g = DiffusionGenerator(; dp = true, epochs = 2, batch_size = 1024,
                               d_layers = [256, 256], num_timesteps = 100)
        push!(cs, Case("diffusion/fit_dpsgd_2ep", "diffusion",
            () -> fit(g, TBL_MED; privacy = PrivacyBudget(epsilon = 10.0), rng = RNG()),
            reps = 3))
    end

    # Evaluation metrics — cheap individually, but `compare` calls them once
    # per engine per seed, so they multiply.
    push!(cs, Case("eval/fidelity_score", "evaluate",
        () -> fidelity_score(TBL_MED, SYN_MED)))
    push!(cs, Case("eval/jensen_shannon", "evaluate",
        () -> jensen_shannon(TBL_MED, SYN_MED)))
    push!(cs, Case("eval/pairwise_marginal_error", "evaluate",
        () -> pairwise_marginal_error(TBL_MED, SYN_MED)))
    push!(cs, Case("eval/utility_tstr", "evaluate",
        () -> utility_tstr(TBL_MED, SYN_MED, :cat2; rng = RNG()), reps = 3))
    # DCR is O(n_synth x n_real), so it is measured on the small table.
    push!(cs, Case("eval/privacy_dcr_2k", "evaluate",
        () -> privacy_dcr(TBL_SMALL, mixed_table(2_000; seed = 5)), reps = 3))

    # Internals with a history of being accidentally quadratic or sync-bound.
    let Ext = Base.get_extension(DataMimic, :DataMimicLuxExt)
        if Ext !== nothing
            v = randn(Float32, 200_000); ref = sort(v)
            push!(cs, Case("internal/quantile_forward_200k", "internal",
                () -> begin
                    s = 0f0
                    for x in v; s += Ext._quantile_forward(x, ref, 200_000); end
                    s
                end, reps = 3))
            push!(cs, Case("internal/rdp_accountant_x65", "internal",
                () -> for _ in 1:65; Ext._rdp_accountant(1.0, 0.06, 1000, 1e-5); end))
        end
    end

    return cs
end

# ── Baseline ───────────────────────────────────────────────────────────────

machine_id() = string(Sys.CPU_NAME, "/", Sys.MACHINE, "/", DEVICE_NAME)

function load_baseline()
    isfile(BASELINE_PATH) || return nothing
    try
        TOML.parsefile(BASELINE_PATH)
    catch e
        @warn "Could not read baseline" exception = e
        nothing
    end
end

function save_baseline(results)
    data = Dict{String,Any}(
        "meta" => Dict{String,Any}(
            "machine"      => machine_id(),
            "device"       => DEVICE_NAME,
            "julia"        => string(VERSION),
            "recorded"     => string(now()),
        ),
        "cases" => Dict{String,Any}(
            name => Dict{String,Any}("min" => lo, "median" => med)
            for (name, med, lo, _) in results
        ),
    )
    open(BASELINE_PATH, "w") do io
        TOML.print(io, data)
    end
    println("\nBaseline written to $(relpath(BASELINE_PATH))")
    println("Commit it so later runs have something to compare against.")
end

# ── Main ───────────────────────────────────────────────────────────────────

function main()
    cases = build_cases()
    RUN_SLOW || (cases = filter(c -> !c.slow, cases))
    isempty(FILTER) || (cases = filter(c -> occursin(FILTER, c.name), cases))

    if isempty(cases)
        println("No cases matched.")
        return 0
    end

    println("DataMimic performance suite")
    println("  device : $DEVICE_NAME")
    println("  machine: $(machine_id())")
    println("  julia  : $VERSION")
    println()

    baseline = load_baseline()
    base_cases = baseline === nothing ? Dict{String,Any}() : get(baseline, "cases", Dict())
    comparable = baseline !== nothing &&
                 get(get(baseline, "meta", Dict()), "machine", "") == machine_id()
    if baseline !== nothing && !comparable
        println("Baseline was recorded on a different machine/device")
        println("  baseline: $(get(get(baseline,"meta",Dict()), "machine", "?"))")
        println("  current : $(machine_id())")
        println("Timings below are reported without comparison.\n")
    end

    # Warm every case before timing any of them: per-case warmup does not
    # cover process-level lazy initialization (BLAS threads, CUDA context,
    # dispatch caches), which otherwise lands entirely on the first case.
    print("Warming up $(length(cases)) case(s)... ")
    for c in cases; c.f(); end
    sync()
    println("done
")

    results = Tuple{String,Float64,Float64,Float64}[]
    regressions = String[]
    ratios = Float64[]

    @printf("%-34s %10s %10s %10s   %s\n", "case", "median", "min", "max", "vs baseline")
    println("─"^88)
    group = ""
    for c in cases
        if c.group != group
            group = c.group
            println("· $group")
        end
        med, lo, hi = measure(c)
        push!(results, (c.name, med, lo, hi))

        note = ""
        if comparable && haskey(base_cases, c.name)
            # Compare minima; fall back to an older median-only baseline.
            bc = base_cases[c.name]
            b = Float64(get(bc, "min", get(bc, "median", NaN)))
            ratio = lo / b
            pct = (ratio - 1) * 100
            push!(ratios, ratio)
            if ratio > REGRESS_FACTOR
                note = @sprintf("REGRESSED %+.0f%% (was %.3fs)", pct, b)
                push!(regressions, c.name)
            elseif ratio < IMPROVE_FACTOR
                note = @sprintf("improved %+.0f%% (was %.3fs)", pct, b)
            else
                note = @sprintf("%+.0f%%", pct)
            end
        elseif comparable
            note = "new"
        end
        @printf("  %-32s %9.3fs %9.3fs %9.3fs   %s\n", c.name, med, lo, hi, note)
    end

    # A machine under load slows every case at once. Reporting that as N
    # independent regressions is misleading, so summarize the drift and let
    # the reader judge. Deliberately does not adjust the gate: normalizing
    # away the median would hide a change that genuinely slowed everything.
    if !isempty(ratios)
        drift = median(ratios)
        if drift > 1.10 || drift < 0.90
            println()
            @printf("NOTE: the median case is %+.0f%% against baseline.
", (drift - 1) * 100)
            println("A uniform shift across unrelated cases usually means machine state")
            println("(load, thermal, power profile) rather than a code change. Re-run on")
            println("an idle machine before believing any individual regression below.")
        end
    end

    println()
    if SAVE_BASE
        save_baseline(results)
        return 0
    end

    if baseline === nothing
        println("No baseline found. Record one with:")
        println("  julia --project=benchmark benchmark/perf.jl --save")
        return 0
    end
    if !comparable
        return 0
    end
    if isempty(regressions)
        println("No regressions (threshold: $(REGRESS_FACTOR)x).")
        return 0
    end
    println("$(length(regressions)) regression(s) past $(REGRESS_FACTOR)x:")
    for r in regressions
        println("  - $r")
    end
    return 1
end

exit(main())
