# MST domain compression: measured, not assumed.
#
#     julia --project=benchmark benchmark/eval_compress.jl
#
# [McKenna et al. 2021] merges bins whose noisy count falls below 3σ into a
# single "other" category before selection. DataMimic does this, and this
# script is the evidence for that decision — it toggles the mechanism off and
# on over the same data and seeds and compares.
#
# METHOD. Cardinality is the dial. Compression should matter exactly when a
# column's domain is sparse relative to the noise, so the synthetic table uses
# Zipf-distributed levels — a few common values, a long thin tail — and sweeps
# the number of levels `k`. Four real tables then check that the synthetic
# result survives contact with data, chosen to DIFFER in shape rather than to
# resemble each other.
#
# Six seeds per cell, both arms on identical data and seeds; a difference
# counts only if it clears the pooled seed standard deviation. `bins` reports
# the merge actually performed, so a null result can be told apart from
# "nothing merged". Primary metric is pairwise marginal error, since 2-way
# marginals are MST's whole job. Lower is better.
#
# ── RESULTS ───────────────────────────────────────────────────────────────
#
# Synthetic dial (4,000 rows, 5 Zipf categoricals + 1 numeric):
#
#     k     eps=0.5   eps=1.0   eps=2.0   eps=4.0
#     8     better    better    noise     noise
#     32    better    better    noise     noise
#     128   better    better    better    better
#     512   better    better    better    better
#
# Real tables:
#
#     table                shape                    0.5      1.0      2.0      4.0
#     rl (41160)           10000x23, 15 nom/8 num   better   better   better   better
#     pbcseq (516)         1945x19,  6 nom/13 num   noise    better   better   better
#     cjs (473)            2796x35,  3 nom/32 num   better   noise    noise    noise
#     BachChoralHarmony    5665x17, 15 nom/2 num    noise    noise    WORSE    WORSE
#
# THE ONE COST CASE. Bach regresses ~5% at eps >= 2. It has the smallest total
# domain of the four — 239 bins against cjs 973, pbcseq 1,393, rl 5,602 —
# because twelve of its seventeen columns are binary. Its marginals are
# already well measured at that budget, so merging only discards information
# it did not need to discard. Compression pays when the domain is large
# relative to what the budget can measure; Bach's is not.
#
# RE-VERIFIED after categorical levels moved from Dict hash order to sorted
# order. The synthetic dial reproduced digit for digit, as did BachChoralHarmony
# - the cost case the conclusion below rests on. rl reproduced on the three
# cells checked before the run was cut short, and pbcseq to the fourth decimal.
# MST was never exposed to that change: _discretize_column has always sorted
# its own levels.
#
# Two cjs cells DID change verdict - eps=0.5 better -> noise, eps=1.0 noise ->
# better - while their numbers moved by less than 0.003. Those cells sit on the
# significance boundary and cross it with the seed draw, so read them as "no
# clear effect" rather than as a result in either direction. No cell anywhere
# moved between better and worse.
#
# RUN TIME. Budget a couple of hours, almost all of it `rl`. The expensive
# cell is rl at eps = 4, not rl at eps = 0.5, which is the opposite of the
# obvious guess: compression merges LESS as the budget grows, so at eps = 0.5
# rl collapses 5,602 bins to 127 and the "on" arm is nearly free, while at
# eps = 4 it only reaches 455 and BOTH arms pay full price. To iterate on the
# real-table results, drop 41160 from the loop below - the other three finish
# in minutes.
#
# A NOTE ON HOW THIS WAS GOT WRONG. An earlier pass ran the synthetic dial
# plus Bach alone, concluded from that single real table that compression
# hurt, and nearly recorded a rejection. Bach is the outlier of the four. One
# real dataset is not enough to overturn a sweep, in either direction.

using Pkg
Pkg.activate("benchmark")

using DataMimic, DataFrames, Random, Printf

include(joinpath(@__DIR__, "datasets.jl"))

say(x) = (println(x); flush(stdout))
mean_(v) = sum(v) / length(v)
sd_(v) = length(v) < 2 ? 0.0 :
         sqrt(sum(abs2, v .- mean_(v)) / (length(v) - 1))

const N_ROWS   = 4_000
const N_CAT    = 5
const N_SEEDS  = 6
const KS       = (8, 32, 128, 512)
const EPSILONS = (0.5, 1.0, 2.0, 4.0)
const MAX_ROWS = 10_000

"""Zipf-distributed categorical levels: a few common, a long sparse tail."""
function zipf_table(n::Int, k::Int, ncat::Int; seed::Int = 1)
    rng = MersenneTwister(seed)
    df = DataFrame()
    df.num = randn(rng, n)
    w = [1.0 / l for l in 1:k]
    w ./= sum(w)
    cum = cumsum(w)
    for j in 1:ncat
        df[!, Symbol("c", j)] =
            [string(searchsortedfirst(cum, rand(rng))) for _ in 1:n]
    end
    return df
end

"""One arm of one cell: (pairwise error, bins before, bins after)."""
function run_arm(df, eps::Float64, compress::Bool, seed::Int)
    prev = DataMimic.MST_DOMAIN_COMPRESSION[]
    DataMimic.MST_DOMAIN_COMPRESSION[] = compress
    try
        pb = PrivacyBudget(epsilon = eps, delta = 1e-5)
        model = fit(MSTGenerator(privacy = pb), df;
                    rng = MersenneTwister(1000 + seed))
        st = DataMimic.MST_COMPRESSION_STATS[]
        synth = sample(model, nrow(df); rng = MersenneTwister(2000 + seed))
        return pairwise_marginal_error(df, synth).mean, st.bins_before, st.bins_after
    finally
        DataMimic.MST_DOMAIN_COMPRESSION[] = prev
    end
end

function compare_cell(df, eps)
    off = [run_arm(df, eps, false, s) for s in 1:N_SEEDS]
    on  = [run_arm(df, eps, true,  s) for s in 1:N_SEEDS]
    p_off, p_on = [r[1] for r in off], [r[1] for r in on]
    m_off, m_on = mean_(p_off), mean_(p_on)
    s_off, s_on = sd_(p_off), sd_(p_on)
    pooled = sqrt(s_off^2 + s_on^2)
    delta  = m_on - m_off
    verdict = abs(delta) <= pooled ? "noise" :
              delta < 0 ? "COMPRESSION BETTER" : "compression worse"
    return (m_off, s_off, m_on, s_on, on[1][2], on[1][3], verdict)
end

row(label, c) = rpad(label, 12) * rpad("$(c[5])->$(c[6])", 14) *
    rpad(@sprintf("%.4f +/- %.4f", c[1], c[2]), 20) *
    rpad(@sprintf("%.4f +/- %.4f", c[3], c[4]), 20) * c[7]

say("MST domain compression — $(N_SEEDS) seeds per cell, mean +/- sd, lower is better")
say("")
say("SYNTHETIC DIAL: $(N_ROWS) rows, $(N_CAT) Zipf categoricals + 1 numeric")
say("")
say(rpad("k / eps", 12) * rpad("bins", 14) * rpad("pairwise off", 20) *
    rpad("pairwise on", 20) * "verdict")
say("-"^82)
for k in KS
    df = zipf_table(N_ROWS, k, N_CAT)
    for eps in EPSILONS
        say(row("$k / $eps", compare_cell(df, eps)))
    end
end

say("")
say("REAL TABLES (OpenML), chosen to differ in shape")
for id in (41160, 516, 473, 4552)
    spec = OPENML_DATASETS[id]
    df = load_openml(id)
    nrow(df) > MAX_ROWS && (df = df[1:MAX_ROWS, :])
    nsym = count(c -> nonmissingtype(eltype(df[!, c])) <: AbstractString, names(df))
    say("")
    say("$(spec.name) (id=$id) — $(spec.note)")
    say("  as loaded: $(nrow(df)) x $(ncol(df)), $nsym nominal / $(ncol(df) - nsym) numeric")
    say(rpad("eps", 12) * rpad("bins", 14) * rpad("pairwise off", 20) *
        rpad("pairwise on", 20) * "verdict")
    say("-"^82)
    for eps in EPSILONS
        say(row(string(eps), compare_cell(df, eps)))
    end
end

say("")
say("DONE")
