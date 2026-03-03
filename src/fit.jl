using DataFrames
using StatsBase
using Copulas

# ─── Marginal fitting ────────────────────────────────────────────────────────

# nm       — pre-collected non-missing values (avoids a redundant column scan)
# col_type — detected column type
# T        — non-missing eltype of the original column
function _fit_marginal(nm::Vector, col_type::Symbol, T::Type)
    if col_type == :constant
        val = isempty(nm) ? missing : first(nm)
        return ConstantMarginal(val)

    elseif col_type in (:continuous, :integer)
        # sort! on the broadcasted copy — one allocation, sorted in-place.
        sorted = sort!(Float64.(nm))
        return EmpiricalMarginal(sorted, T)

    else  # :categorical or :binary
        # countmap is a single O(n) pass; the old [count(==(l), nm) for l in lvls]
        # was O(n × k) where k is the number of unique levels.
        cm    = StatsBase.countmap(nm)
        lvls  = collect(keys(cm))
        probs = Float64.(collect(values(cm)))
        probs ./= sum(probs)
        return CategoricalMarginal(lvls, probs)
    end
end

# ─── Main fit function ────────────────────────────────────────────────────────

"""
    fit(df::DataFrame; scramble=Symbol[]) -> SynthModel

Profile `df` and return a fitted `SynthModel`. No synthetic data is produced.

Columns named in `scramble` are sampled normally but have their characters
(strings) or digits (integers) randomly scrambled before appearing in the
synthetic output, so no real identifier value is ever directly exposed.
"""
function fit(df::DataFrame; scramble::Vector{Symbol} = Symbol[])
    nrows, ncols = size(df)
    nrows == 0 && error("Input DataFrame has zero rows.")
    ncols == 0 && error("Input DataFrame has zero columns.")

    col_names   = Symbol.(names(df))
    col_types   = Symbol[]
    marginals   = Dict{Symbol, Any}()
    missingness = Dict{Symbol, Float64}()

    # Validate scramble list up front.
    unknown = setdiff(scramble, col_names)
    isempty(unknown) || error("Columns not found in DataFrame: $unknown")

    for name in col_names
        col = df[!, name]
        n   = length(col)

        # Collect non-missing values once and derive everything from that.
        nm     = _nonmissing(col)
        p_miss = (n - length(nm)) / n   # no extra scan over the column
        missingness[name] = p_miss

        if p_miss == 1.0
            @warn "Column $name is entirely missing; treating as Constant(missing)."
        end

        T     = _basetype(col)
        ctype = _detect_column_type(nm, T)   # reuses nm
        push!(col_types, ctype)
        marginals[name] = _fit_marginal(nm, ctype, T)   # reuses nm
    end

    # Warn if a scrambled column has a type that scrambling won't meaningfully obscure.
    for cname in scramble
        idx   = findfirst(==(cname), col_names)
        ctype = col_types[idx]
        if ctype == :constant
            @warn "Scrambled column $cname is :constant (single value); " *
                  "scrambling may not change the output."
        elseif ctype == :continuous
            @warn "Scrambled column $cname is :continuous (Float64/Float32); " *
                  "digit-scrambling of floats may produce unexpected values."
        end
    end

    # Identify numeric columns for the copula
    numeric_mask = [t in (:continuous, :integer) for t in col_types]
    copula_cols  = col_names[numeric_mask]

    copula = _fit_copula(df, col_names, col_types, copula_cols)

    return SynthModel(col_names, col_types, marginals, missingness,
                      copula, copula_cols, nrows, scramble)
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
