using DataFrames
using StatsBase
using Copulas

# ─── Marginal fitting ────────────────────────────────────────────────────────

function _fit_marginal(col, col_type::Symbol)
    nm = _nonmissing(col)

    if col_type == :constant
        val = isempty(nm) ? missing : first(nm)
        return ConstantMarginal(val)

    elseif col_type in (:continuous, :integer)
        sorted = sort(Float64.(nm))
        return EmpiricalMarginal(sorted, _basetype(col))

    else  # :categorical or :binary
        lvls = unique(nm)
        counts = [count(==(l), nm) for l in lvls]
        probs = counts ./ sum(counts)
        return CategoricalMarginal(lvls, Float64.(probs))
    end
end

# ─── Main fit function ────────────────────────────────────────────────────────

"""
    fit(df::DataFrame) -> SynthModel

Profile `df` and return a fitted `SynthModel`. No synthetic data is produced.
"""
function fit(df::DataFrame)
    nrows, ncols = size(df)
    nrows == 0 && error("Input DataFrame has zero rows.")
    ncols == 0 && error("Input DataFrame has zero columns.")

    col_names  = Symbol.(names(df))
    col_types  = Symbol[]
    marginals  = Dict{Symbol, Any}()
    missingness = Dict{Symbol, Float64}()

    for name in col_names
        col = df[!, name]
        n   = length(col)
        p_miss = count(ismissing, col) / n
        missingness[name] = p_miss

        if p_miss == 1.0
            @warn "Column $name is entirely missing; treating as Constant(missing)."
        end

        ctype = detect_column_type(col)
        push!(col_types, ctype)
        marginals[name] = _fit_marginal(col, ctype)
    end

    # Identify numeric columns for the copula
    numeric_mask = [t in (:continuous, :integer) for t in col_types]
    copula_cols  = col_names[numeric_mask]

    copula = _fit_copula(df, col_names, col_types, copula_cols)

    return SynthModel(col_names, col_types, marginals, missingness,
                      copula, copula_cols, nrows)
end

# ─── Copula fitting ───────────────────────────────────────────────────────────

function _fit_copula(df, col_names, col_types, copula_cols)
    d = length(copula_cols)

    if d == 0
        @warn "No numeric columns found; all columns are categorical/constant. " *
              "Falling back to fully independent sampling."
        return nothing
    end

    if d == 1
        @warn "Only one numeric column present; copula fitting skipped."
        return nothing
    end

    # Build a Float64 matrix (n_rows × d) with NaN for missing values.
    n = nrow(df)
    X = Matrix{Float64}(undef, n, d)
    for (j, cname) in enumerate(copula_cols)
        col = df[!, cname]
        for i in 1:n
            v = col[i]
            X[i, j] = ismissing(v) ? NaN : Float64(v)
        end
    end

    # Complete cases across all numeric columns.
    complete = vec(.!any(isnan.(X), dims=2))
    Xc = X[complete, :]   # (n_complete × d)

    if size(Xc, 1) < 2
        @warn "Fewer than 2 complete cases for copula fitting; using independent sampling."
        return nothing
    end

    # Copulas.jl convention: data matrix is (d × n_obs).
    # pseudos computes scaled ranks column-wise (per variable).
    U = Copulas.pseudos(Matrix(Xc'))   # (d × n_complete)

    return StatsBase.fit(BetaCopula, U)
end
