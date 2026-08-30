using Pkg
Pkg.activate("benchmark")

using DataMimic, Lux, LuxCUDA, Zygote, CSV, DataFrames, Random

const TARGET = Symbol(" <=50K")

println("Loading data...")
df = CSV.read("benchmark/data/adult_train.csv", DataFrame)
dropmissing!(df)
println("Train: $(nrow(df)) rows")

df_test = CSV.read("benchmark/data/adult_test.csv", DataFrame;
                    header=names(df), skipto=2)
dropmissing!(df_test)
# Fix trailing period on target
df_test[!, TARGET] = replace.(df_test[!, TARGET], r"\.$" => "")
println("Test: $(nrow(df_test)) rows")

# TabDDPM's tuned Adult configuration (exp/adult/ddpm_cb_best/config.toml):
#   d_layers = [256, 1024, 1024, 1024, 1024, 256], dropout = 0.0
#   num_timesteps = 100, steps = 30000, lr = 0.00201, weight_decay = 0.0,
#   batch_size = 4096, is_y_cond = true, normalization = "quantile"
# 30000 steps ÷ ceil(32560/4096) = 8 batches/epoch ⇒ 3750 epochs.

println("\n=== Training: TabDDPM tuned Adult config (class-conditional, 30k steps) ===")
gen = DiffusionGenerator(
    epochs        = 3750,
    batch_size    = 4096,
    d_layers      = [256, 1024, 1024, 1024, 1024, 256],
    num_timesteps = 100,
    dropout       = 0.0,
    lr            = 0.0020099410620098234,
    weight_decay  = 0.0,
    lr_warmup     = 0,
    target        = TARGET,
)
model = fit(gen, df; rng=Random.MersenneTwister(42))

println("\n=== Sampling ===")
synth = sample(model, nrow(df))
println("Synth: $(nrow(synth)) rows")

println("\n=== Fidelity ===")
fid = fidelity_score(df, synth)
println("  Aggregate: $(round(fid.aggregate; digits=4))")
println("  Correlation: $(round(fid.correlation_score; digits=4))")

println("\n=== TSTR (EvoTrees, F1, held-out test set) ===")
tstr = utility_tstr(df, synth, TARGET; test=df_test)
println("  Task: $(tstr.task)")
println("  Synth F1:    $(round(tstr.synth_score; digits=4))")
println("  Real F1:     $(round(tstr.real_score; digits=4))")
println("  F1 Ratio:    $(round(tstr.ratio; digits=4))")
println("  Synth Acc:   $(round(tstr.synth_accuracy; digits=4))")
println("  Real Acc:    $(round(tstr.real_accuracy; digits=4))")

println("\n=== TSTR (no test set, internal 80/20 split) ===")
tstr2 = utility_tstr(df, synth, TARGET; rng=MersenneTwister(42))
println("  Synth F1:    $(round(tstr2.synth_score; digits=4))")
println("  Real F1:     $(round(tstr2.real_score; digits=4))")
println("  F1 Ratio:    $(round(tstr2.ratio; digits=4))")

println("\nDone!")
