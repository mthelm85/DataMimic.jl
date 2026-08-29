using Pkg
Pkg.activate("benchmark")

using DataMimic, DataFrames, Lux, Zygote, LuxCUDA, Random

println("=== DDIM Step Skipping Test ===\n")

# Create test data
rng = MersenneTwister(42)
n = 500
df = DataFrame(
    age = rand(rng, 20:70, n),
    income = randn(rng, n) .* 10000 .+ 50000,
    hours = rand(rng, 10:60, n),
    workclass = rand(rng, ["Private", "Self-emp", "Gov", "Other", "Without-pay"], n),
    education = rand(rng, ["HS", "Some-college", "Bachelors", "Masters", "PhD", "Assoc"], n),
    marital = rand(rng, ["Married", "Single", "Divorced", "Widowed", "Separated"], n),
    sex = rand(rng, ["Male", "Female"], n)
)
println("Data: $(nrow(df)) rows, $(ncol(df)) cols")

println("\n--- Fitting (5 epochs) ---")
t0 = time()
model = DataMimic.fit(DiffusionGenerator(epochs=5, batch_size=128), df; rng=MersenneTwister(123))
println("Fit: $(round(time() - t0, digits=1))s")

n_sample = 500

# Warm up
_ = DataMimic.sample(model, 10; rng=MersenneTwister(1))

# Full 1000 steps
println("\n--- Sampling $n_sample rows ---")
t1 = time()
synth_full = DataMimic.sample(model, n_sample; rng=MersenneTwister(42))
println("Full (1000 steps): $(round(time() - t1, digits=2))s")

# 200 steps
t2 = time()
synth_200 = DataMimic.sample(model, n_sample; rng=MersenneTwister(42), sampling_steps=200)
println("DDIM (200 steps):  $(round(time() - t2, digits=2))s")

# 100 steps
t3 = time()
synth_100 = DataMimic.sample(model, n_sample; rng=MersenneTwister(42), sampling_steps=100)
println("DDIM (100 steps):  $(round(time() - t3, digits=2))s")

# 50 steps
t4 = time()
synth_50 = DataMimic.sample(model, n_sample; rng=MersenneTwister(42), sampling_steps=50)
println("DDIM (50 steps):   $(round(time() - t4, digits=2))s")

# 25 steps
t5 = time()
synth_25 = DataMimic.sample(model, n_sample; rng=MersenneTwister(42), sampling_steps=25)
println("DDIM (25 steps):   $(round(time() - t5, digits=2))s")

# Sanity check all outputs
println("\n--- Sanity checks ---")
for (label, synth) in [("Full", synth_full), ("200", synth_200),
                        ("100", synth_100), ("50", synth_50), ("25", synth_25)]
    wc = length(unique(synth.workclass))
    ed = length(unique(synth.education))
    sx = length(unique(synth.sex))
    println("  $label: workclass=$wc, education=$ed, sex=$sx unique values")
end

# With eta=1.0 (stochastic DDIM ≈ DDPM)
t6 = time()
synth_eta1 = DataMimic.sample(model, n_sample; rng=MersenneTwister(42),
                               sampling_steps=100, ddim_eta=1.0)
println("\nDDIM (100 steps, η=1.0): $(round(time() - t6, digits=2))s")
wc = length(unique(synth_eta1.workclass))
println("  workclass=$wc unique values")

println("\n=== DONE ===")
