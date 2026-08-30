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

### Deterministic Sampling (DDIM)

- **Song, J., Meng, C., & Ermon, S. (2020).**
  *Denoising Diffusion Implicit Models.*
  ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

  Deterministic reverse-process sampler enabling fewer sampling steps
  (`sampling_steps` parameter) without retraining. η parameter controls
  stochasticity (η=0 → deterministic DDIM, η=1 → DDPM).

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
  arXiv:1702.07476v3.

  Exact privacy accounting via subsampled Rényi divergence. Computes
  the (ε, δ) guarantee for the noise multiplier and number of training
  steps.

---

## Copula Generator

### Gaussian Copula

- **Nelsen, R. B. (2006).**
  *An Introduction to Copulas.* 2nd edition. Springer.

  General copula theory. The `CopulaGenerator` uses a Gaussian copula
  (Spearman rank → Pearson conversion) to model column dependencies.

---

## MST Generator

### Private Marginal Selection

- **McKenna, R., Miklau, G., & Sheldon, D. (2021).**
  *Winning the NIST Contest: A scalable and general approach to
  differentially private synthetic data.*
  Journal of Privacy and Confidentiality, 11(3).

  MST algorithm: exponential-mechanism marginal selection → Gaussian-noise
  measurement → PGM reconstruction. Used in `MSTGenerator`.

### Private Graphical Models

- **McKenna, R., Sheldon, D., & Miklau, G. (2019).**
  *Graphical-model based estimation and inference for differential privacy.*
  ICML 2019.

  PGM inference engine used inside MST for reconstructing a full joint
  distribution from noisy marginal measurements.

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
