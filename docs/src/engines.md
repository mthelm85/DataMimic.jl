# Engines

| Generator | Private | Reach for it when |
|---|---|---|
| [`CopulaGenerator`](@ref) | no | You want speed. Strong on mixed numeric/categorical tables |
| [`DiffusionGenerator`](@ref) | optional | You want the highest fidelity and can afford training time |
| [`MSTGenerator`](@ref) | yes | Private, categorical-heavy data |
| [`DPCopulaGenerator`](@ref) | yes | Private, continuous-heavy data |

Each engine implements a published algorithm. Where the paper and its reference
implementation disagree, DataMimic follows the implementation, and the
differences that remain are noted per engine below.

## Choosing between them

The table above narrows the field but will not pick a winner. Relative
performance depends on properties of the data that its shape does not capture,
and for the private engines it also moves with ε and with row count — a private
engine improves substantially as rows increase at fixed ε, because its noise is
fixed while the signal grows.

[`compare`](@ref) settles it empirically: it fits a list of engines to your own
table, repeats each over several seeds, and reports the mean and spread of
whatever metrics you name. See [Comparing engines](evaluation.md#Comparing-engines).

---

## CopulaGenerator

```julia
CopulaGenerator()            # :beta, the default
CopulaGenerator(:gaussian)
```

A copula separates a joint distribution into two independent parts: the
marginal distribution of each column, and the dependence structure linking
them. Sklar's theorem guarantees this decomposition always exists, which makes
it a natural fit for tabular data — each column's shape is modelled exactly by
its empirical distribution, and only the coupling between columns needs to be
estimated.

**How it works.** Each column is mapped to a uniform pseudo-observation on
[0, 1] by its own empirical CDF. A copula is fitted to those uniforms. To
sample, DataMimic draws from the copula and pushes each coordinate back through
the corresponding inverse CDF, which reproduces the original marginals by
construction.

**Categorical columns** take part in the copula through the *distributional
transform*: a level occupying the CDF interval `[F(k-1), F(k)]` maps to a
uniform draw inside that interval, and sampling inverts the same step function.
Dependence between categorical and numeric columns is therefore modelled rather
than discarded.

Two consequences are worth knowing. The association a copula can express is
monotone in the level ordering, which is arbitrary for a nominal variable, so
this captures a real part of the dependence but not all of it. And the two
copula families differ sharply here: `:beta` is nonparametric and can represent
the non-monotone structure an arbitrary ordering produces, while `:gaussian` is
restricted to a single correlation matrix and cannot. On the Adult dataset the
train-on-synthetic utility ratio is about 0.99 for `:beta` against 0.66 for
`:gaussian`. Prefer the default unless you specifically want a Gaussian
dependence structure.

`:gaussian` carries one further restriction. It estimates a correlation matrix
and factorizes it, and that factorization fails when the modelled columns are
linearly dependent after the rank transform. Fewer complete cases than columns
guarantees it; collinear columns can, though whether an exactly duplicated
column trips it is a matter of round-off. DataMimic adjusts such a matrix to
the nearest positive-definite correlation matrix and warns, rather than
failing: a near-duplicate pair comes back at a correlation indistinguishable
from 1, but dependence among the affected columns is approximate. `:beta`
needs no such adjustment.

A categorical column with only one observed level cannot be encoded this way,
so it is left out of the copula and drawn independently.

**Cost.** Fitting is a rank transform plus a copula fit — seconds on tables
where the diffusion model takes many minutes. This is the engine to start with.

**References.** [Sklar 1959]; Nelsen, *An Introduction to Copulas*, 2nd ed.
(2006). Copula fitting is delegated to
[Copulas.jl](https://github.com/lrnv/Copulas.jl).

---

## DiffusionGenerator

```julia
using Lux, Zygote      # activates the extension

DiffusionGenerator(; epochs = 100, batch_size = 512, target = nothing)
```

TabDDPM adapted to Julia. A diffusion model learns to reverse a noising
process: noise is added to real rows over many timesteps until nothing remains,
and a network is trained to undo one step at a time. Generating a row means
starting from pure noise and running that learned reversal.

Tabular data needs two noising processes at once, and TabDDPM runs them in
parallel over the same timestep:

- **Numeric columns** get *Gaussian diffusion* — the standard DDPM process,
  where the network predicts the noise that was added.
- **Categorical columns** get *multinomial diffusion*, which interpolates each
  one-hot vector toward the uniform distribution over its levels. This is
  carried in log space and trained against the variational bound, so
  categorical structure is modelled directly rather than being embedded into a
  continuous space and rounded back.

Numeric columns are first mapped to an approximately Gaussian shape by a
quantile transform, matching the reference pipeline; heavy tails and skew
otherwise dominate the objective.

**Class-conditional generation.** Naming a `target` column conditions the
denoiser on an embedding of the label and draws labels from the empirical class
distribution at sampling time. On classification-style tables this markedly
improves downstream utility, and it reproduces the paper's own setup:

```julia
gen = DiffusionGenerator(
    epochs        = 3750,
    batch_size    = 4096,
    d_layers      = [256, 1024, 1024, 1024, 1024, 256],
    num_timesteps = 100,
    target        = :income_bracket,
)
model = fit(gen, df)
```

**Tuning.** This engine has far more configuration than the others, and its
results depend on it. The defaults are deliberately modest so a first run
finishes quickly; they are not tuned for quality. `d_layers` sets explicit
per-layer widths and overrides `hidden_dim`/`n_blocks`, `num_timesteps` sets
the length of the diffusion process, and `ema_decay` controls the exponential
moving average of weights used for sampling. The configuration above is the
paper's tuned Adult architecture and is a reasonable starting point for a table
of similar size.

An undertrained diffusion model can score no better than independent sampling.
If it places last in a comparison, check the epoch count before concluding
anything about the method.

**Divergence.** Training aborts with an explanatory error if the loss stops
being finite, rather than burning the remaining epochs on weights that can no
longer recover. A lower `lr`, a smaller `batch_size`, or a narrower `d_layers`
usually resolves it.

**Privacy.** `dp = true` trains with DP-SGD and requires a budget; see
[Privacy](privacy.md).

DP training is slower than ordinary training but no longer dramatically so.
Clipping each example's gradient individually appears to require one backward
pass per example; it does not. For a `Dense` layer, example *i*'s weight
gradient is an outer product, so its norm is the product of two column norms —
which means every example's norm, and the clipped sum, come out of two batched
passes rather than *B* unbatched ones. This is *ghost clipping* (Goodfellow
2015; Li et al. 2021), and it is an exact identity rather than an
approximation: DataMimic's tests assert the two agree to floating-point
rounding, both per step and over a full training run.

Larger batches make DP training *cheaper* per epoch, as they do for ordinary
training — the opposite of the per-example behaviour.

**References.** Kotelnikov et al., *TabDDPM* (ICML 2023,
[arXiv:2209.15421](https://arxiv.org/abs/2209.15421)), cross-checked against
[yandex-research/tab-ddpm](https://github.com/yandex-research/tab-ddpm);
Ho et al., *DDPM* (NeurIPS 2020); Hoogeboom et al., *Multinomial Diffusion*
(NeurIPS 2021); Nichol & Dhariwal (ICML 2021) for the cosine noise schedule.

---

## MSTGenerator

```julia
MSTGenerator()      # 2-way marginals
```

MST won the NIST differential privacy synthetic data challenge, and it is
built around a simple observation: you cannot afford to measure everything
privately, so spend the budget on the relationships that matter most.

**How it works.**

1. **Discretize.** Every column is binned (numeric columns into 32 equal-width
   bins by default), because the algorithm works over contingency tables.
2. **Measure all one-way marginals** under Gaussian noise. These are cheap —
   there are only as many as there are columns — and they anchor every
   column's own distribution.
3. **Select a spanning tree.** Each candidate pair of columns is scored by how
   badly the independence assumption fails for it, and a maximum spanning tree
   over columns is chosen using the exponential mechanism, which makes the
   selection itself private. A tree, rather than an arbitrary graph, is what
   keeps the next step exact.
4. **Measure the selected two-way marginals** under Gaussian noise.
5. **Reconcile.** The noisy measurements are mutually inconsistent — one
   marginal's implied column totals will not match another's. Private-PGM
   fits a single tree-structured graphical model that best explains all of
   them at once, by entropic mirror descent over the marginal polytope with
   exact belief propagation for inference.
6. **Sample** ancestrally from the reconciled model, then map bins back to
   values.

Reconciliation is pure post-processing — it touches only already-private
measurements — so it costs no budget. It matters most at tight budgets, where
measurement noise dominates and the inconsistencies are large; by around ε = 4
it makes little practical difference.

**Budget split.** 30% to tree selection, 20% to the one-way marginals, 50% to
the selected two-way marginals, composed under zCDP.

**Where it fits.** Because everything is discretized, MST is at its best on
categorical or naturally discrete data and loses information on continuous
columns through binning. It also benefits more than the public engines from
extra rows at fixed ε.

**Gaps.** Domain compression — merging low-count bins before selection — is the
one part of the published algorithm not implemented. `max_marginal_order = 3`
is accepted but warns and falls back to 2-way marginals.

**References.** McKenna, Miklau & Sheldon, *Winning the NIST Contest* (JPC
11(3), 2021); McKenna, Sheldon & Miklau, *Graphical-model based estimation and
inference for differential privacy* (ICML 2019). Cross-checked against
[ryan112358/private-pgm](https://github.com/ryan112358/private-pgm).

---

## DPCopulaGenerator

```julia
DPCopulaGenerator()
```

The copula idea again, with both halves made private: differentially private
histogram marginals combined with a differentially private covariance matrix,
assembled into a Gaussian copula.

**How it works.** Each column's marginal is estimated as a histogram with
Gaussian noise added to the counts. The dependence structure comes from a
private covariance matrix estimated by the Analyze-Gauss mechanism, which adds
symmetric Gaussian noise to the second-moment matrix — symmetric because the
matrix is symmetric, and noising the two triangles independently would give the
off-diagonal entries half the intended variance. Sampling draws from the
resulting Gaussian copula and inverts the noisy marginals.

**Budget split.** Half to the marginals, half to the covariance, composed under
zCDP.

**Where it fits.** A covariance matrix is a natural summary of continuous data
and a poor one for nominal categories, so this is the private engine for
continuous-heavy tables — the mirror image of MST. It is also comparatively
insensitive to row count, which makes it competitive on smaller tables where
MST is still noise-dominated.

**References.** Dwork, Talwar, Thakurta & Zhang, *Analyze Gauss* (STOC 2014);
Nelsen (2006) for the copula construction.

---

## Reference notes

The full bibliography, including the specific points where this package departs
from a paper, is in
[`references/REFERENCES.md`](https://github.com/mthelm85/DataMimic.jl/blob/main/references/REFERENCES.md).
