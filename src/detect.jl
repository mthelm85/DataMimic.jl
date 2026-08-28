# ─── Column Type Detection ───────────────────────────────────────────────────

"""Return the non-missing element type of a column."""
_basetype(col) = nonmissingtype(eltype(col))

"""
Collect all non-missing, finite values from a column.

For numeric columns this also drops `NaN` and `Inf`/`-Inf` — those
values carry no statistical signal and would corrupt the empirical
marginal.  For non-numeric types the filter is a no-op.
"""
function _nonmissing(col)
    T = nonmissingtype(eltype(col))
    if T <: AbstractFloat
        return [x for x in col if !ismissing(x) && isfinite(x)]
    else
        return collect(skipmissing(col))
    end
end

"""
    detect_column_type(col) -> Symbol

Classify a column into one of:
`:constant`, `:binary`, `:continuous`, `:integer`, `:categorical`.

Detection is **type-aware**: `Bool` is always `:binary`; low-cardinality
integers (≤ `min(20, 5% of n)` unique values) are treated as `:categorical`
rather than `:integer`, since they usually represent encoded categories.
"""
function detect_column_type(col)
    nm = _nonmissing(col)
    return _detect_column_type(nm, _basetype(col))
end

# ─── Internal detection ─────────────────────────────────────────────────────

function _detect_column_type(nm::Vector, T::Type)
    isempty(nm) && return :constant

    # Bool is always binary — the eltype itself signals a two-level
    # variable, even when only one level appears in the sample.
    T <: Bool && return :binary

    n       = length(nm)
    n_unique = length(unique(nm))

    n_unique == 1 && return :constant

    # Strings / Symbols
    if T <: AbstractString || T <: Symbol
        n_unique == 2 && return :binary
        return :categorical
    end

    # Floats
    if T <: AbstractFloat
        is_whole = all(x -> x == floor(x), nm)
        if is_whole
            return _classify_numeric_cardinality(n_unique, n)
        end
        n_unique == 2 && return :binary
        return :continuous
    end

    # Integers
    if T <: Integer
        return _classify_numeric_cardinality(n_unique, n)
    end

    # Fallback for anything else (Date, Char, custom types, Any, …)
    n_unique == 2 && return :binary
    return :categorical
end

"""
Decide whether a numeric column (integer or whole-number float) with
`n_unique` distinct values out of `n` observations is `:binary`,
`:categorical`, or `:integer`.

Threshold: `min(20, max(2, n ÷ 20))`.  This means
  • For tiny samples (n < 60), only ≤ 2 unique values → categorical.
  • For larger samples, the 5%-of-n rule scales up but caps at 20.
  • 20 is a pragmatic ceiling — columns like "day of month" (≤ 31)
    are borderline, but anything with fewer than 20 levels is rarely
    a meaningful continuous quantity.
"""
function _classify_numeric_cardinality(n_unique::Int, n::Int)
    n_unique == 2 && return :binary
    threshold = min(20, max(2, n ÷ 20))
    n_unique ≤ threshold && return :categorical
    return :integer
end
