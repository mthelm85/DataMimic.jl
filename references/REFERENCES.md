# DataMimic.jl — Key References

Papers and methods underpinning each engine and evaluation module.

---

## Diffusion Generator (TabDDPM)

### Core Architecture

- **Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023).**
  *TabDDPM: Modelling Tabular Data with Diffusion Models.*
  ICML 2023. [arXiv:2209.15421](https://arxiv.org/abs/2209.15421)

  Primary reference for the `DiffusionGenerator` engine. Contributions used:
  - `MLPDiffusion` backbone: a plain MLP (`Dense → ReLU → Dropout` blocks, no
    normalization or residual connections) with the sinusoidal timestep
    embedding added once at the input projection
  - Class-conditional generation: `silu(label_emb(y))` added to the timestep
    embedding, with labels drawn from the empirical class distribution at
    sampling time
  - Multinomial diffusion for categorical features, carried in log space and
    trained against the stochastic variational bound
  - Gaussian quantile normalization for continuous features
  - AdamW with linear learning-rate annealing and an EMA of the denoiser weights
  - TSTR (Train on Synthetic, Test on Real) evaluation protocol with CatBoost + macro-F1

  Implementation cross-checked against the reference code at
  [yandex-research/tab-ddpm](https://github.com/yandex-research/tab-ddpm),
  including the tuned Adult configuration used in `benchmark/eval_tstr.jl`.

### Cosine Noise Schedule

- **Nichol, A., & Dhariwal, P. (2021).**
  *Improved Denoising Diffusion Probabilistic Models.*
  ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

  Cosine β schedule, ᾱ(t) = cos²((t + 0.008)/1.008 · π/2), used as TabDDPM's
  default noise schedule.

### Tabular MLP / ResNet Baselines

- **Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).**
  *Revisiting Deep Learning Models for Tabular Data.*
  NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)

  The `rtdl` baseline MLP that TabDDPM uses as its denoising network.

### Gaussian Diffusion

- **Ho, J., Jain, A., & Abbeel, P. (2020).**
  *Denoising Diffusion Probabilistic Models.*
  NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

  Linear β schedule, forward noising process q(x_t | x_0), simplified
  training objective (predict noise ε).

### Deterministic Sampling (DDIM) — *not implemented*

- **Song, J., Meng, C., & Ermon, S. (2020).**
  *Denoising Diffusion Implicit Models.*
  ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

  A deterministic reverse-process sampler that allows fewer sampling steps
  without retraining. **`DiffusionGenerator` does not implement this** — it
  samples with the full stochastic DDPM reverse process over `num_timesteps`
  steps. Listed as the obvious route to faster sampling, not as a description
  of current behaviour.

### Multinomial Diffusion

- **Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P., & Welling, M. (2021).**
  *Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions.*
  NeurIPS 2021. [arXiv:2102.05379](https://arxiv.org/abs/2102.05379)

  Uniform-noise diffusion on one-hot categorical vectors. Used for all
  categorical/binary columns in `DiffusionGenerator`.

---

## Differential Privacy

### DP-SGD

- **Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I.,
  Talwar, K., & Zhang, L. (2016).**
  *Deep Learning with Differential Privacy.*
  CCS 2016. [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)

  Per-example gradient clipping + Gaussian noise addition. Used when
  `DiffusionGenerator(dp=true)` is set.

### Rényi DP Accounting

- **Mironov, I., Talwar, K., & Zhang, L. (2019).**
  *Rényi Differential Privacy of the Sampled Gaussian Mechanism.*
  arXiv:1908.10530.

  Closed-form Rényi divergence of the Poisson-subsampled Gaussian at
  integer orders — exact per order, but the reported (ε, δ) for a given
  noise multiplier and step count is an upper bound, since the order
  search is over a finite integer grid and the RDP → (ε, δ) conversion
  is the standard [Mironov 2017] one.

---

## Copula Generator

### Gaussian Copula

- **Nelsen, R. B. (2006).**
  *An Introduction to Copulas.* 2nd edition. Springer.

  General copula theory. `CopulaGenerator` models dependence with a Beta
  (default) or Gaussian copula fitted by `Copulas.jl` to rank-based
  pseudo-observations.

  Categorical and binary columns **are** part of the copula, encoded by the
  distributional transform: a level occupying the empirical CDF interval
  `[F(k-1), F(k)]` maps to a uniform draw inside that interval, and sampling
  inverts the same step function. Including them raised train-on-synthetic
  utility on Adult from 0.54 to 0.99. The association this can express is
  monotone in the level ordering, which is arbitrary for a nominal variable,
  so it captures a substantial part of the dependence rather than all of it —
  and the nonparametric Beta copula handles the resulting non-monotone
  structure far better than the Gaussian one (0.99 against 0.66 on Adult).
  A categorical column with a single observed level is left out and drawn
  independently.

---

## MST Generator

### Private Marginal Selection

- **McKenna, R., Miklau, G., & Sheldon, D. (2021).**
  *Winning the NIST Contest: A scalable and general approach to
  differentially private synthetic data.*
  Journal of Privacy and Confidentiality, 11(3).

  MST algorithm: exponential-mechanism marginal selection → Gaussian-noise
  measurement → PGM reconstruction.

  Implemented by `MSTGenerator`: all 1-way marginals measured, spanning-tree
  selection by exponential mechanism scored on count-scale L1 error against the
  independence reference, Gaussian-noise measurement of the selected 2-way
  marginals, then Private-PGM reconciliation before ancestral sampling.
  Cross-checked against the reference implementation at
  [ryan112358/private-pgm](https://github.com/ryan112358/private-pgm)
  (`mechanisms/mst.py`). Domain compression is the one remaining gap; see the
  MST implementation note in REQUIREMENTS.md §11.

### Private Graphical Models

- **McKenna, R., Sheldon, D., & Miklau, G. (2019).**
  *Graphical-model based estimation and inference for differential privacy.*
  ICML 2019.

  The PGM inference engine that MST uses to reconcile noisy marginal
  measurements into a consistent joint distribution. Implemented in
  `src/engines/mst.jl` as entropic mirror descent over the marginal polytope,
  with exact sum-product belief propagation for the inference step — the model
  is a spanning tree, so two passes suffice and no general junction-tree
  machinery is required.

### Higher-Order Marginals (AIM) — *not implemented*

- **McKenna, R., Mullins, B., Sheldon, D., & Miklau, G. (2022).**
  *AIM: An Adaptive and Iterative Mechanism for Differentially Private
  Synthetic Data.*
  PVLDB 15(11): 2599–2612.
  [arXiv:2201.12677](https://arxiv.org/abs/2201.12677)

  The successor to MST, and the reason `MSTGenerator` has no marginal-order
  setting. MST fixes the structure to a spanning tree over 2-way marginals;
  AIM selects marginals of varying order adaptively, budgeting each round
  against how much the measurement is expected to help. Higher-order marginals
  are not a parameter you can turn on in MST — the tree is what makes belief
  propagation exact, and abandoning it means junction-tree inference and a
  different budget argument. **Not implemented**; listed as the published route
  to higher-order structure, not as a description of current behaviour.

---

## Evaluation

### TSTR Protocol

- **Esteban, C., Hyland, S. L., & Rätsch, G. (2017).**
  *Real-valued (Medical) Time Series Generation with Recurrent
  Conditional GANs.* arXiv:1706.02633.

  Train-on-Synthetic-Test-on-Real protocol, adapted for tabular data.
  `utility_tstr` implements this with gradient-boosted trees (EvoTrees.jl)
  and macro-averaged F1, following TabDDPM's evaluation setup.

### Gaussian Quantile Normalization

- **Scikit-learn QuantileTransformer.**
  [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)

  Maps features to a Gaussian distribution via empirical CDF → Φ⁻¹.
  Used in preprocessing for `DiffusionGenerator`, matching TabDDPM's
  data pipeline.
