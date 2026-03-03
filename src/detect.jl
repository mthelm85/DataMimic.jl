# Returns the non-missing eltype of a column.
_basetype(col) = nonmissingtype(eltype(col))

# Returns all non-missing values from a column as a Vector.
_nonmissing(col) = collect(skipmissing(col))

"""
    detect_column_type(col) -> Symbol

Classify a DataFrame column into one of:
  :constant, :binary, :continuous, :integer, :categorical
"""
function detect_column_type(col)
    nm = _nonmissing(col)
    return _detect_column_type(nm, _basetype(col))
end

# Internal version — accepts pre-collected non-missing values and base type
# so fit() can pass the nm it already collected instead of re-scanning the column.
function _detect_column_type(nm::Vector, T::Type)
    isempty(nm) && return :constant

    n_unique = length(unique(nm))
    n_unique == 1 && return :constant
    n_unique == 2 && return :binary

    if T <: AbstractFloat
        all(x -> x == floor(x), nm) && return :integer
        return :continuous
    end

    T <: Integer && return :integer

    # String, Symbol, Bool, CategoricalValue → categorical
    return :categorical
end
