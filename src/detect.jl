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

    # Entirely missing → constant
    isempty(nm) && return :constant

    n_unique = length(unique(nm))

    # Priority 1: single unique value
    n_unique == 1 && return :constant

    # Priority 2: exactly two unique values (regardless of type)
    n_unique == 2 && return :binary

    # Priority 3: type-based for 3+ unique values
    T = _basetype(col)

    if T <: AbstractFloat
        # Integer-valued floats → integer
        all(x -> x == floor(x), nm) && return :integer
        return :continuous
    end

    T <: Integer && return :integer

    # String, Symbol, Bool, CategoricalValue → categorical
    return :categorical
end
