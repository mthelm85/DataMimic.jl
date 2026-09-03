# ─── utility_tstr ──────────────────────────────────────────────────────────
#
# REQ-EVL-007: Train on synth, evaluate on held-out real
# REQ-EVL-008: Auto-detect classification vs regression
# REQ-EVL-009: Return accuracy/F1/RMSE for synth-trained and real-trained + ratio
#
# Reference: [Esteban et al. 2017] — TSTR protocol
# Classifier: EvoTrees gradient boosted trees (comparable to CatBoost)
# Metric: Macro-averaged F1 for classification (following TabDDPM [Kotelnikov et al. 2023])

# ─── Helpers ────────────────────────────────────────────────────────────────

"""
Encode feature columns as a numeric matrix for EvoTrees.

- Numeric columns → Float64 as-is (missing → column median).
- Categorical columns → integer codes (missing → mode code).
"""
function _encode_features(cols, feature_names::Vector{Symbol},
                          nrows::Int;
                          ref_cols = nothing  # for consistent encoding
                         )
    ref = isnothing(ref_cols) ? cols : ref_cols
    mat = Matrix{Float64}(undef, nrows, length(feature_names))

    for (j, name) in enumerate(feature_names)
        raw = Tables.getcolumn(cols, name)
        ref_col = Tables.getcolumn(ref, name)
        kind = _eval_column_kind(ref_col)

        if kind == :numeric
            ref_nm = collect(Float64, filter(x -> !ismissing(x) && isfinite(x), ref_col))
            med = isempty(ref_nm) ? 0.0 : StatsBase.median(ref_nm)
            for i in 1:nrows
                v = raw[i]
                mat[i, j] = (ismissing(v) || !isfinite(v)) ? med : Float64(v)
            end
        else
            ref_nm = filter(!ismissing, ref_col)
            levels = unique(ref_nm)
            level_map = Dict(lv => Float64(i) for (i, lv) in enumerate(levels))
            mode_code = isempty(ref_nm) ? 0.0 :
                        level_map[StatsBase.mode(ref_nm)]
            for i in 1:nrows
                v = raw[i]
                mat[i, j] = ismissing(v) ? mode_code :
                            get(level_map, v, 0.0)
            end
        end
    end

    return mat
end

"""
Extract the target vector, classifying the task as `:classification` or
`:regression` based on element type.
"""
function _extract_target(cols, target::Symbol, nrows::Int)
    raw = Tables.getcolumn(cols, target)
    nm  = filter(!ismissing, raw)
    isempty(nm) && throw(ArgumentError("Target column :$target is entirely missing."))

    T = typeof(first(nm))
    task = (T <: AbstractString || T <: Symbol || T <: Bool) ?
           :classification : :regression

    if task == :classification
        labels = Vector{String}(undef, nrows)
        mode_label = string(StatsBase.mode(nm))
        for i in 1:nrows
            v = raw[i]
            labels[i] = ismissing(v) ? mode_label : string(v)
        end
        return labels, task
    else
        vals = Vector{Float64}(undef, nrows)
        med = StatsBase.median(collect(Float64, filter(x -> !ismissing(x) && isfinite(x), nm)))
        for i in 1:nrows
            v = raw[i]
            vals[i] = (ismissing(v) || !isfinite(v)) ? med : Float64(v)
        end
        return vals, task
    end
end

"""
Stratified train/test split: partition row indices so that each class
appears in roughly the same proportion in both sets.
"""
function _stratified_split(y::Vector{String}, test_frac::Float64, rng::AbstractRNG)
    n = length(y)
    classes = unique(y)
    train_idx = Int[]
    test_idx  = Int[]
    for cls in classes
        cls_idx = findall(==(cls), y)
        cls_idx = cls_idx[Random.randperm(rng, length(cls_idx))]
        n_test = max(1, round(Int, length(cls_idx) * test_frac))
        append!(test_idx,  cls_idx[1:n_test])
        append!(train_idx, cls_idx[n_test+1:end])
    end
    return train_idx, test_idx
end

"""
Macro-averaged F1 score: compute per-class precision, recall, F1 and
average across all classes. Returns (f1_macro, accuracy).
"""
function _macro_f1(y_true::Vector{String}, y_pred::Vector{String})
    classes = sort(unique(vcat(y_true, y_pred)))
    n = length(y_true)
    acc = count(y_true .== y_pred) / n

    f1_sum = 0.0
    for cls in classes
        tp = count((y_true .== cls) .& (y_pred .== cls))
        fp = count((y_true .!= cls) .& (y_pred .== cls))
        fn = count((y_true .== cls) .& (y_pred .!= cls))

        precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
        recall    = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
        f1_sum += f1
    end

    f1_macro = f1_sum / length(classes)
    return f1_macro, acc
end

# ─── Public API ─────────────────────────────────────────────────────────────

"""
    utility_tstr(real, synth, target::Symbol;
                 test=nothing, test_frac=0.2, nrounds=200,
                 max_depth=6, eta=0.05, nbins=64,
                 rng=Random.default_rng()) -> NamedTuple

Train-on-Synthetic, Test-on-Real evaluation using gradient boosted trees
from `EvoTrees.jl`.

1. Train a model on `synth` features → `synth` target.
2. Evaluate on held-out real data → real target.
3. Also train on real train data → evaluate on the same held-out set (baseline).

If `test` is provided (a Tables.jl-compatible table), it is used as the
held-out test set. Otherwise, `real` is split into train/test using
`test_frac` (default 0.2) with stratified sampling for classification.

For classification, reports both macro-averaged F1 score (following the
TabDDPM evaluation protocol [Kotelnikov et al. 2023]) and accuracy.
The ratio is computed from F1 scores.

Returns a `NamedTuple` with:
- `task`: `:classification` or `:regression`
- `synth_score`: F1 (classification) or RMSE (regression) of synth-trained model
- `real_score`: F1 / RMSE of real-trained model (baseline)
- `ratio`: `synth_score / real_score` — closer to 1.0 is better
- `synth_accuracy`, `real_accuracy`: (classification only) raw accuracy scores
"""
function utility_tstr(real, synth, target::Symbol;
                      test = nothing,
                      test_frac::Float64 = 0.2,
                      nrounds::Int = 200,
                      max_depth::Int = 6,
                      eta::Float64 = 0.05,
                      nbins::Int = 64,
                      rng::AbstractRNG = Random.default_rng())
    Tables.istable(real)  || throw(ArgumentError("real must be a Tables.jl table"))
    Tables.istable(synth) || throw(ArgumentError("synth must be a Tables.jl table"))
    if test !== nothing
        Tables.istable(test) || throw(ArgumentError("test must be a Tables.jl table"))
    end

    r_cols = Tables.columns(real)
    s_cols = Tables.columns(synth)
    r_names = collect(Symbol, Tables.columnnames(r_cols))
    s_names = collect(Symbol, Tables.columnnames(s_cols))

    target in r_names || throw(ArgumentError(
        "target :$target not found in real table."))
    target in s_names || throw(ArgumentError(
        "target :$target not found in synth table."))

    # Feature columns: shared columns minus target
    shared = sort(collect(Symbol, intersect(Set(r_names), Set(s_names))))
    feature_names = filter(!=(target), shared)
    isempty(feature_names) && throw(ArgumentError(
        "No shared feature columns between real and synth."))

    n_real  = length(Tables.getcolumn(r_cols, first(r_names)))
    n_synth = length(Tables.getcolumn(s_cols, first(s_names)))

    # Encode synth features and target (always uses all synth rows for training)
    X_synth = _encode_features(s_cols, feature_names, n_synth; ref_cols = r_cols)
    y_synth, task = _extract_target(s_cols, target, n_synth)

    if test !== nothing
        # ── External held-out test set ─────────────────────────────────
        t_cols  = Tables.columns(test)
        n_test  = length(Tables.getcolumn(t_cols, first(Tables.columnnames(t_cols))))
        X_train = _encode_features(r_cols, feature_names, n_real; ref_cols = r_cols)
        X_test  = _encode_features(t_cols, feature_names, n_test; ref_cols = r_cols)
        y_train, _ = _extract_target(r_cols, target, n_real)
        y_test, _  = _extract_target(t_cols, target, n_test)
    else
        # ── Internal train/test split ──────────────────────────────────
        X_all  = _encode_features(r_cols, feature_names, n_real; ref_cols = r_cols)
        y_all, _ = _extract_target(r_cols, target, n_real)

        if task == :classification
            train_idx, test_idx = _stratified_split(y_all, test_frac, rng)
        else
            perm = Random.randperm(rng, n_real)
            n_test_split = max(1, round(Int, n_real * test_frac))
            test_idx  = perm[1:n_test_split]
            train_idx = perm[n_test_split+1:end]
        end

        X_train = X_all[train_idx, :]
        X_test  = X_all[test_idx, :]
        y_train = y_all[train_idx]
        y_test  = y_all[test_idx]
    end

    if task == :classification
        # ── Classification with EvoTrees ───────────────────────────────
        config = EvoTrees.EvoTreeClassifier(;
            nrounds, max_depth, eta, nbins,
            early_stopping_rounds = 50)

        # Train on synth, test on held-out real
        m_synth = EvoTrees.fit(config;
            x_train = X_synth, y_train = y_synth,
            x_eval = X_test, y_eval = y_test,
            verbosity = 0)
        pred_probs_synth = EvoTrees.predict(m_synth, X_test)
        levels_synth = m_synth.info[:target_levels]
        preds_synth = [String(levels_synth[argmax(pred_probs_synth[i, :])]) for i in axes(pred_probs_synth, 1)]
        synth_f1, synth_acc = _macro_f1(y_test, preds_synth)

        # Baseline: train on real train, test on held-out real
        m_real = EvoTrees.fit(config;
            x_train = X_train, y_train = y_train,
            x_eval = X_test, y_eval = y_test,
            verbosity = 0)
        pred_probs_real = EvoTrees.predict(m_real, X_test)
        levels_real = m_real.info[:target_levels]
        preds_real = [String(levels_real[argmax(pred_probs_real[i, :])]) for i in axes(pred_probs_real, 1)]
        real_f1, real_acc = _macro_f1(y_test, preds_real)

        ratio = real_f1 > 0 ? synth_f1 / real_f1 : 0.0

        return (;
            task           = task,
            synth_score    = synth_f1,
            real_score     = real_f1,
            ratio          = ratio,
            synth_accuracy = synth_acc,
            real_accuracy  = real_acc,
        )
    else
        # ── Regression with EvoTrees ───────────────────────────────────
        config = EvoTrees.EvoTreeRegressor(;
            nrounds, max_depth, eta, nbins,
            early_stopping_rounds = 50)

        m_synth = EvoTrees.fit(config;
            x_train = X_synth, y_train = Float64.(y_synth),
            x_eval = X_test, y_eval = Float64.(y_test),
            verbosity = 0)
        preds_synth = vec(EvoTrees.predict(m_synth, X_test))
        synth_rmse = sqrt(sum(abs2, preds_synth .- y_test) / length(y_test))

        m_real = EvoTrees.fit(config;
            x_train = X_train, y_train = Float64.(y_train),
            x_eval = X_test, y_eval = Float64.(y_test),
            verbosity = 0)
        preds_real = vec(EvoTrees.predict(m_real, X_test))
        real_rmse = sqrt(sum(abs2, preds_real .- y_test) / length(y_test))

        ratio = real_rmse > 0 ? synth_rmse / real_rmse : Inf

        return (;
            task        = task,
            synth_score = synth_rmse,
            real_score  = real_rmse,
            ratio       = ratio,
        )
    end
end

"""
    utility_tstr(target::Symbol; kwargs...) -> Function

Partially applied form, for `compare`.

`compare` calls each metric as `f(real, synth)`, so a metric needing more than
those two arguments would otherwise have to be wrapped in an anonymous
function. This returns that wrapper:

```julia
compare(generators, df;
        metrics = (fidelity = fidelity_score,
                   utility  = utility_tstr(:income)))
```

Keywords are forwarded, so `utility_tstr(:income; nrounds = 100)` works too.
The result is a `NamedTuple`, and `compare` reads `.ratio` from it without
being told to - see `metric_field`.

!!! note "Passing `rng` here captures one generator"
    The returned closure holds whatever `rng` you give it and advances it on
    every call, so `utility_tstr(:income; rng = MersenneTwister(1))` does not
    give each of `compare`'s seeds the same train/test split - it gives them
    successive draws from one stream. That is usually what you want. To fix
    the split across every call, build a fresh generator inside a lambda
    instead: `(r, s) -> utility_tstr(r, s, :income; rng = MersenneTwister(1))`.
"""
utility_tstr(target::Symbol; kwargs...) =
    (real, synth) -> utility_tstr(real, synth, target; kwargs...)
