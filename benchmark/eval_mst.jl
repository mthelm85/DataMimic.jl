# MSTGenerator utility sweep on Adult.
#
#     julia --project=benchmark benchmark/eval_mst.jl
#
# Baseline for the current implementation (root 1-way marginal only,
# mutual-information edge scoring, row-normalized conditionals):
#
#     eps    fidelity   corr     TSTR F1   ratio
#     0.5    0.1642     0.0969   0.6370    0.7950
#     1.0    0.1388     0.0805   0.6318    0.7886
#     2.0    0.1222     0.0714   0.6330    0.7901
#     4.0    0.1150     0.0702   0.6353    0.7929
#     8.0    0.1117     0.0699   0.6278    0.7836
#
# Note the TSTR ratio is flat in epsilon while fidelity improves.  That is a
# symptom, not a coincidence: tree selection is effectively a uniform random
# draw at these budgets (see the measured-limitation note under REQ-MST-007),
# so the captured dependence structure does not improve as epsilon grows.
#
# Any change to the engine should be compared against the table above before
# landing.  A Private-PGM reconciliation prototype improved fidelity to ~0.110
# across the sweep but dropped the TSTR ratio to ~0.68 and was rejected on that
# basis; it is preserved at dev/mst-pgm-wip.patch.

using Pkg
Pkg.activate("benchmark")
say(x) = (println(x); flush(stdout))

using DataMimic, CSV, DataFrames, Random

const TARGET = Symbol(" <=50K")

df = CSV.read("benchmark/data/adult_train.csv", DataFrame)
dropmissing!(df)
df_test = CSV.read("benchmark/data/adult_test.csv", DataFrame;
                   header = names(df), skipto = 2)
dropmissing!(df_test)
df_test[!, TARGET] = replace.(df_test[!, TARGET], r"\.$" => "")

say("Adult: $(nrow(df)) train / $(nrow(df_test)) test rows")
say("")
say(rpad("epsilon", 10) * rpad("fidelity", 12) * rpad("corr", 12) *
    rpad("TSTR F1", 12) * rpad("ratio", 10) * "fit(s)")
say("-"^62)

for eps in (0.5, 1.0, 2.0, 4.0, 8.0)
    pb = PrivacyBudget(epsilon = eps, delta = 1e-5)
    t0 = time()
    model = fit(MSTGenerator(), df; privacy = pb, rng = MersenneTwister(42))
    t_fit = time() - t0

    synth = sample(model, nrow(df); rng = MersenneTwister(7))
    fid   = fidelity_score(df, synth)
    tstr  = utility_tstr(df, synth, TARGET; test = df_test)

    say(rpad(eps, 10) *
        rpad(round(fid.aggregate; digits = 4), 12) *
        rpad(round(fid.correlation_score; digits = 4), 12) *
        rpad(round(tstr.synth_score; digits = 4), 12) *
        rpad(round(tstr.ratio; digits = 4), 10) *
        string(round(t_fit; digits = 1)))
end

say("")
say("(fidelity and corr: lower is better; TSTR ratio: higher is better)")
say("DONE")
