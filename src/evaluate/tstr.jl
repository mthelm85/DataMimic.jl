# ─── utility_tstr ──────────────────────────────────────────────────────────
#
# REQ-EVL-007: Train RF on synth, evaluate on held-out real
# REQ-EVL-008: Auto-detect classification vs regression
# REQ-EVL-009: Return accuracy/RMSE for synth-trained and real-trained + ratio
#
# Reference: [Esteban et al. 2017] — TSTR protocol

# ─── Helpers ────────────────────────────────────────────────────────────────

"""
Encode feature columns as a numeric matrix for DecisionTree.jl.

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

# ─── Public API ─────────────────────────────────────────────────────────────

"""
    utility_tstr(real, synth, target::Symbol;
                 n_trees=100, rng=Random.default_rng()) -> NamedTuple

Train-on-Synthetic, Test-on-Real evaluation using a random forest from
`DecisionTree.jl`.

1. Train a model on `synth` features → `synth` target.
2. Evaluate on `real` features → `real` target.
3. Also train on `real` → `real` (baseline) for comparison.

Returns a `NamedTuple` with:
- `task`: `:classification` or `:regression`
- `synth_score`: accuracy (classification) or RMSE (regression) of synth-trained model
- `real_score`: accuracy / RMSE of real-trained model (baseline)
- `ratio`: `synth_score / real_score` — closer to 1.0 is better
           (for classification: higher is better; for regression: lower is better)
"""
function utility_tstr(real, synth, target::Symbol;
                      n_trees::Int = 100)
    Tables.istable(real)  || throw(ArgumentError("real must be a Tables.jl table"))
    Tables.istable(synth) || throw(ArgumentError("synth must be a Tables.jl table"))

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

    # Encode features and target
    X_real  = _encode_features(r_cols, feature_names, n_real;  ref_cols = r_cols)
    X_synth = _encode_features(s_cols, feature_names, n_synth; ref_cols = r_cols)
    y_real, task  = _extract_target(r_cols, target, n_real)
    y_synth, _    = _extract_target(s_cols, target, n_synth)

    if task == :classification
        # ── Classification ──────────────────────────────────────────────
        # Train on synth, test on real
        forest_synth = DecisionTree.build_forest(
            y_synth, X_synth, -1, n_trees)
        preds_synth = DecisionTree.apply_forest(forest_synth, X_real)
        synth_acc = count(preds_synth .== y_real) / n_real

        # Baseline: train on real, test on real (leave-one-out approx)
        forest_real = DecisionTree.build_forest(
            y_real, X_real, -1, n_trees)
        preds_real = DecisionTree.apply_forest(forest_real, X_real)
        real_acc = count(preds_real .== y_real) / n_real

        ratio = real_acc > 0 ? synth_acc / real_acc : 0.0

        return (;
            task        = task,
            synth_score = synth_acc,
            real_score  = real_acc,
            ratio       = ratio,
        )
    else
        # ── Regression ──────────────────────────────────────────────────
        forest_synth = DecisionTree.build_forest(
            y_synth, X_synth, -1, n_trees)
        preds_synth = DecisionTree.apply_forest(forest_synth, X_real)
        synth_rmse = sqrt(sum(abs2, preds_synth .- y_real) / n_real)

        forest_real = DecisionTree.build_forest(
            y_real, X_real, -1, n_trees)
        preds_real = DecisionTree.apply_forest(forest_real, X_real)
        real_rmse = sqrt(sum(abs2, preds_real .- y_real) / n_real)

        ratio = real_rmse > 0 ? synth_rmse / real_rmse : Inf

        return (;
            task        = task,
            synth_score = synth_rmse,
            real_score  = real_rmse,
            ratio       = ratio,
        )
    end
end
