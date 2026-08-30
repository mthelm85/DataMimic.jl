# CopulaGenerator fidelity on Adult, with attention to the categorical columns.
#
#     julia --project=benchmark benchmark/eval_copula.jl
#
# `pairwise_marginal_error` is the metric that sees categorical dependence: it
# compares 2-way marginals across ALL column pairs, including
# categorical-categorical and categorical-numeric, which `fidelity_score`'s
# correlation term (numeric-only Spearman) cannot.
#
# Before categoricals were included in the copula (5 seeds):
#
#     engine      fidelity            pairwise error      TSTR ratio
#     :beta       0.0296 +/- 0.0009   0.0780 +/- 0.0002   0.5410 +/- 0.0013
#     :gaussian   0.0586 +/- 0.0013   0.0799 +/- 0.0004   0.5425 +/- 0.0027
#
# After:
#
#     engine      fidelity            pairwise error      TSTR ratio
#     :beta       0.0046 +/- 0.0004   0.0132 +/- 0.0005   0.9923 +/- 0.0023
#     :gaussian   0.0323 +/- 0.0001   0.0715 +/- 0.0001   0.6573 +/- 0.0126
#
# The gap between :beta and :gaussian is the point of interest: a Gaussian
# copula can only express monotone rank dependence, and the ordinal encoding of
# a nominal column is arbitrary, so much of the structure is non-monotone.
# BetaCopula is nonparametric and captures it.

using Pkg
Pkg.activate("benchmark")
say(x) = (println(x); flush(stdout))
using DataMimic, CSV, DataFrames, Random

const TARGET = Symbol(" <=50K")
df = CSV.read("benchmark/data/adult_train.csv", DataFrame); dropmissing!(df)
df_test = CSV.read("benchmark/data/adult_test.csv", DataFrame;
                   header = names(df), skipto = 2)
dropmissing!(df_test)
df_test[!, TARGET] = replace.(df_test[!, TARGET], r"\.$" => "")
say("Adult: $(nrow(df)) train rows, $(ncol(df)) columns")

mean_(v) = sum(v) / length(v)
sd_(v)   = length(v) < 2 ? 0.0 : sqrt(sum(abs2, v .- mean_(v)) / (length(v) - 1))

const NSEEDS = 5

for ct in (:beta, :gaussian)
    fid, pme, tstr = Float64[], Float64[], Float64[]
    t0 = time()
    for seed in 1:NSEEDS
        m = fit(CopulaGenerator(ct), df; rng = MersenneTwister(seed))
        s = sample(m, nrow(df); rng = MersenneTwister(1000 + seed))
        push!(fid,  fidelity_score(df, s).aggregate)
        push!(pme,  pairwise_marginal_error(df, s).mean)
        push!(tstr, utility_tstr(df, s, TARGET; test = df_test).ratio)
    end
    say("")
    say("CopulaGenerator(:$ct)   [$(round((time()-t0)/NSEEDS; digits=2))s per fit+sample]")
    say("  fidelity aggregate      $(round(mean_(fid);  digits=4)) ± $(round(sd_(fid);  digits=4))")
    say("  pairwise marginal error $(round(mean_(pme);  digits=4)) ± $(round(sd_(pme);  digits=4))")
    say("  TSTR ratio              $(round(mean_(tstr); digits=4)) ± $(round(sd_(tstr); digits=4))")
end

say("")
say("(fidelity and pairwise error: lower better; TSTR: higher better)")
say("DONE")
