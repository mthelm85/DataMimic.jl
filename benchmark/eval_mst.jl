# MSTGenerator utility sweep on Adult.
#
#     julia --project=benchmark benchmark/eval_mst.jl
#
# SINGLE-SEED NUMBERS AT LOW EPSILON ARE NOISE.  Measured across 6 seeds:
#
#     eps = 0.5   TSTR ratio  mean 0.707  sd 0.063  range [0.579, 0.744]
#     eps = 2.0   TSTR ratio  mean 0.811  sd 0.015  range [0.796, 0.840]
#
# At eps = 0.5 the seed-to-seed spread is larger than most differences worth
# arguing about, so compare distributions over several seeds there, never two
# single runs.  By eps = 2 the noise collapses and single runs are informative.
#
# Before count-scale edge scoring (mutual-information score, root 1-way
# marginal only, half the budget on selection):
#
#     eps    fidelity   corr     TSTR F1   ratio
#     0.5    0.1642     0.0969   0.6370    0.7950
#     1.0    0.1388     0.0805   0.6318    0.7886
#     2.0    0.1222     0.0714   0.6330    0.7901
#     4.0    0.1150     0.0702   0.6353    0.7929
#     8.0    0.1117     0.0699   0.6278    0.7836
#
# After (count-scale L1 score anchored on noisy 1-way marginals, 30/20/50
# budget split):
#
#     eps    fidelity   corr     TSTR F1   ratio
#     0.5    0.1475     0.0789   0.6146    0.7671
#     1.0    0.1204     0.0717   0.6381    0.7964
#     2.0    0.1138     0.0691   0.6445    0.8044
#     4.0    0.1080     0.0636   0.6413    0.8005
#     8.0    0.1061     0.0627   0.6404    0.7993
#
# Current, with domain compression added (single seed, like the tables above):
#
#     eps    fidelity   corr     TSTR F1   ratio
#     0.5    0.1096     0.0687   0.6119    0.7638
#     1.0    0.1090     0.0688   0.6529    0.8149
#     2.0    0.1090     0.0696   0.6473    0.8079
#     4.0    0.1079     0.0680   0.6467    0.8072
#     8.0    0.1077     0.0677   0.6455    0.8057
#
# Compression buys fidelity where the budget is tight - 0.1475 -> 0.1096 at
# eps = 0.5, a 26% improvement - and changes almost nothing by eps = 8. That
# is the shape to expect: it trades resolution you could not measure for a
# count you can, and at a generous budget you could already measure it. The
# TSTR differences here sit inside the seed noise this header warns about and
# should NOT be read as real; only the fidelity column moves enough to argue
# about from a single seed. See benchmark/eval_compress.jl for the multi-seed
# comparison.
#
# Note that the TSTR ratio now RISES with epsilon (0.767 -> 0.799) where before
# it was flat at ~0.79.  Flatness was the symptom of selection being a uniform
# random draw: extra budget bought nothing.  The eps = 0.5 rows differ by less
# than one seed-standard-deviation and should not be read as a regression.
#
# Private-PGM reconciliation, measured over 6 seeds per cell on top of the
# count-scale selection (mean +/- sd):
#
#     eps   PGM   fidelity            TSTR ratio
#     0.5   off   0.1513 +/- 0.0100   0.7272 +/- 0.0644
#     0.5   on    0.1089 +/- 0.0016   0.7638 +/- 0.0291
#     1.0   off   0.1224 +/- 0.0017   0.7705 +/- 0.0355
#     1.0   on    0.1077 +/- 0.0007   0.7846 +/- 0.0485
#     2.0   off   0.1124 +/- 0.0012   0.8120 +/- 0.0203
#     2.0   on    0.1079 +/- 0.0009   0.8169 +/- 0.0214
#     4.0   off   0.1078 +/- 0.0009   0.8137 +/- 0.0146
#     4.0   on    0.1077 +/- 0.0008   0.8137 +/- 0.0226
#     8.0   off   0.1057 +/- 0.0005   0.8101 +/- 0.0180
#     8.0   on    0.1077 +/- 0.0008   0.8122 +/- 0.0159
#
# The benefit scales inversely with the budget, as a variance-reduction step
# should.  At eps = 8 fidelity is marginally worse with reconciliation: with
# near-exact measurements the binding constraint is tree-model misspecification
# rather than noise.
#
# An earlier version of the same estimation code regressed TSTR by ~0.10 and
# was rejected.  That was measured against random tree selection; once
# selection responded to the budget the regression disappeared entirely.
# Reconciliation cannot help when the tree it propagates over is arbitrary.

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
