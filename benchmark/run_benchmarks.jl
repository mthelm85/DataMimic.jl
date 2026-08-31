# ─── DataMimic Benchmark Suite ────────────────────────────────────────
#
# Run all engines against standard datasets and report fidelity, privacy,
# and utility metrics.
#
# Usage:
#   cd benchmark
#   julia --project=. run_benchmarks.jl
#
# Or from the REPL:
#   ] activate benchmark
#   include("benchmark/run_benchmarks.jl")

using DataMimic
using DataFrames
using Dates
using Random
using Printf
using Lux, Zygote, LuxCUDA

include("datasets.jl")

# ─── Configuration ─────────────────────────────────────────────────────────

const SEED = 42
const RESULTS_DIR = joinpath(@__DIR__, "results")

function ensure_results_dir()
    isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)
end

# ─── Pretty printing ──────────────────────────────────────────────────────

function print_header(title::String)
    println("\n", "="^70)
    println("  ", title)
    println("="^70)
end

function print_section(title::String)
    println("\n  ── ", title, " ", "─"^max(1, 55 - length(title)))
end

function print_metric(label::String, value)
    @printf("    %-35s %.4f\n", label, value)
end

# ─── Run evaluation suite on one (real, synth) pair ────────────────────────

function evaluate_all(real_df, synth_df, target::Symbol)
    results = Dict{String, Any}()

    # Fidelity
    fs = fidelity_score(real_df, synth_df)
    results["fidelity_aggregate"]    = fs.aggregate
    results["fidelity_correlation"]  = fs.correlation_score
    results["fidelity_1d_mean"]      = sum(values(fs.column_scores)) /
                                        max(length(fs.column_scores), 1)

    # Jensen-Shannon
    js = jensen_shannon(real_df, synth_df)
    results["jsd_mean"] = js.mean

    # Pairwise marginal error (2-way)
    # Limit columns to avoid combinatorial explosion on Covertype
    shared = sort(collect(Symbol, intersect(
        Set(propertynames(real_df)), Set(propertynames(synth_df)))))
    if length(shared) ≤ 30
        pme = pairwise_marginal_error(real_df, synth_df; order = 2)
        results["pme_mean"]        = pme.mean
        results["pme_worst_score"] = pme.worst_score
        results["pme_worst_pair"]  = string(pme.worst_pair)
        results["pme_n_pairs"]     = pme.n_pairs
    else
        # Too many columns — sample 20 for PME
        subset_cols = shared[1:20]
        r_sub = select(real_df,  subset_cols)
        s_sub = select(synth_df, subset_cols)
        pme = pairwise_marginal_error(r_sub, s_sub; order = 2)
        results["pme_mean"]        = pme.mean
        results["pme_worst_score"] = pme.worst_score
        results["pme_worst_pair"]  = string(pme.worst_pair)
        results["pme_n_pairs"]     = pme.n_pairs
    end

    # DCR
    dcr = privacy_dcr(real_df, synth_df)
    results["dcr_median"]  = dcr.median
    results["dcr_p5"]      = dcr.p5
    results["dcr_exact"]   = dcr.exact_matches

    # TSTR
    tstr = utility_tstr(real_df, synth_df, target; n_trees = 50)
    results["tstr_task"]       = string(tstr.task)
    results["tstr_synth"]      = tstr.synth_score
    results["tstr_real"]       = tstr.real_score
    results["tstr_ratio"]      = tstr.ratio

    return results
end

function print_results(results::Dict)
    print_section("Fidelity")
    print_metric("Aggregate score",        results["fidelity_aggregate"])
    print_metric("1D marginal mean",       results["fidelity_1d_mean"])
    print_metric("Correlation Frobenius",  results["fidelity_correlation"])

    print_section("Jensen–Shannon Divergence")
    print_metric("Mean JSD",               results["jsd_mean"])

    print_section("Pairwise Marginal Error (2-way)")
    print_metric("Mean TVD",               results["pme_mean"])
    print_metric("Worst pair TVD",         results["pme_worst_score"])
    @printf("    %-35s %s\n", "Worst pair", results["pme_worst_pair"])
    @printf("    %-35s %d\n", "Number of pairs", results["pme_n_pairs"])

    print_section("Privacy — DCR")
    print_metric("Median DCR",            results["dcr_median"])
    print_metric("5th percentile DCR",    results["dcr_p5"])
    @printf("    %-35s %d\n", "Exact matches (DCR=0)", results["dcr_exact"])

    print_section("Utility — TSTR")
    @printf("    %-35s %s\n", "Task", results["tstr_task"])
    if results["tstr_task"] == "classification"
        print_metric("Synth-trained accuracy",  results["tstr_synth"])
        print_metric("Real-trained accuracy",   results["tstr_real"])
    else
        print_metric("Synth-trained RMSE",      results["tstr_synth"])
        print_metric("Real-trained RMSE",       results["tstr_real"])
    end
    print_metric("Ratio (synth/real)",     results["tstr_ratio"])
end

# ─── Benchmark: Adult ──────────────────────────────────────────────────────

function benchmark_adult()
    print_header("BENCHMARK: Adult (Census Income)")

    df = load_adult()
    # Drop rows with missing values for cleaner benchmarking
    dropmissing!(df)
    @info "Adult after dropping missings" rows=nrow(df)

    target = :income
    rng = MersenneTwister(SEED)
    n = nrow(df)

    # ── CopulaGenerator ────────────────────────────────────────────────
    print_header("Adult × CopulaGenerator(:beta)")
    t = @elapsed begin
        model = fit(CopulaGenerator(:beta), df; rng = copy(rng))
        synth = DataMimic.sample(model, n)
    end
    @printf("  Fit + sample time: %.1fs\n", t)
    results = evaluate_all(df, synth, target)
    print_results(results)

    # ── MSTGenerator (ε = 1.0) ─────────────────────────────────────────
    print_header("Adult × MSTGenerator (ε = 1.0)")
    t = @elapsed begin
        model = fit(MSTGenerator(), df;
                    privacy = PrivacyBudget(; epsilon = 1.0),
                    rng = copy(rng))
        synth = DataMimic.sample(model, n)
    end
    @printf("  Fit + sample time: %.1fs\n", t)
    results = evaluate_all(df, synth, target)
    print_results(results)

    # ── DPCopulaGenerator (ε = 1.0) ───────────────────────────────────
    print_header("Adult × DPCopulaGenerator (ε = 1.0)")
    t = @elapsed begin
        model = fit(DPCopulaGenerator(), df;
                    privacy = PrivacyBudget(; epsilon = 1.0),
                    rng = copy(rng))
        synth = DataMimic.sample(model, n)
    end
    @printf("  Fit + sample time: %.1fs\n", t)
    results = evaluate_all(df, synth, target)
    print_results(results)

    # ── DiffusionGenerator (non-private, small epochs for speed) ──────
    print_header("Adult × DiffusionGenerator (non-private, 50 epochs)")
    t = @elapsed begin
        model = fit(DiffusionGenerator(; epochs = 50), df; rng = copy(rng))
        synth = DataMimic.sample(model, n)
    end
    @printf("  Fit + sample time: %.1fs\n", t)
    results = evaluate_all(df, synth, target)
    print_results(results)

    # ── Privacy-Utility Sweep (MSTGenerator) ──────────────────────────
    print_header("Adult × MSTGenerator — Privacy-Utility Sweep")
    epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
    sweep = privacy_utility_sweep(
        MSTGenerator(), df, epsilons, fidelity_score;
        rng = copy(rng))

    println("\n  ε         │ Fidelity Aggregate")
    println("  ──────────┼────────────────────")
    for r in sweep
        @printf("  ε = %5.1f  │ %.4f\n", r.epsilon, r.metric_result.aggregate)
    end
end

# ─── Benchmark: Covertype ─────────────────────────────────────────────────

function benchmark_covertype()
    print_header("BENCHMARK: Covertype (Forest Cover Type)")

    # Subsample to 10k rows for tractability
    df = load_covertype(; n = 10_000)
    target = :cover_type

    # Convert cover_type to String for classification
    df.cover_type = string.(df.cover_type)

    rng = MersenneTwister(SEED)
    n = nrow(df)

    # ── CopulaGenerator ────────────────────────────────────────────────
    print_header("Covertype(10k) × CopulaGenerator(:beta)")
    t = @elapsed begin
        model = fit(CopulaGenerator(:beta), df; rng = copy(rng))
        synth = DataMimic.sample(model, n)
    end
    @printf("  Fit + sample time: %.1fs\n", t)
    results = evaluate_all(df, synth, target)
    print_results(results)

    # ── MSTGenerator (ε = 1.0) ─────────────────────────────────────────
    print_header("Covertype(10k) × MSTGenerator (ε = 1.0)")
    t = @elapsed begin
        model = fit(MSTGenerator(), df;
                    privacy = PrivacyBudget(; epsilon = 1.0),
                    rng = copy(rng))
        synth = DataMimic.sample(model, n)
    end
    @printf("  Fit + sample time: %.1fs\n", t)
    results = evaluate_all(df, synth, target)
    print_results(results)

    # ── DPCopulaGenerator (ε = 1.0) ───────────────────────────────────
    print_header("Covertype(10k) × DPCopulaGenerator (ε = 1.0)")
    t = @elapsed begin
        model = fit(DPCopulaGenerator(), df;
                    privacy = PrivacyBudget(; epsilon = 1.0),
                    rng = copy(rng))
        synth = DataMimic.sample(model, n)
    end
    @printf("  Fit + sample time: %.1fs\n", t)
    results = evaluate_all(df, synth, target)
    print_results(results)

    # ── DiffusionGenerator (non-private, 30 epochs) ───────────────────
    print_header("Covertype(10k) × DiffusionGenerator (non-private, 30 epochs)")
    t = @elapsed begin
        model = fit(DiffusionGenerator(; epochs = 30), df; rng = copy(rng))
        synth = DataMimic.sample(model, n)
    end
    @printf("  Fit + sample time: %.1fs\n", t)
    results = evaluate_all(df, synth, target)
    print_results(results)
end

# ─── Main ──────────────────────────────────────────────────────────────────

function main()
    ensure_results_dir()

    println()
    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║             DataMimic — Benchmark Suite                          ║")
    println("║           $(Dates.now())                                        ║")
    println("╚══════════════════════════════════════════════════════════════════╝")

    benchmark_adult()
    benchmark_covertype()

    println("\n\n✅ All benchmarks complete.")
end

# Only run if executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
