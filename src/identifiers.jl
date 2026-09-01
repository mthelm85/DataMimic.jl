# ─── Identifier Detection & Fill ────────────────────────────────────────────

# Minimum non-missing observations before the auto-detection heuristic may
# call a column an identifier. Below this, all-distinct says nothing: any
# column looks like a key when you have only a few values of it.
const MIN_IDENTIFIER_OBSERVATIONS = 20

"""
    _resolve_identifiers(...) -> Set{Symbol}

Determine the full set of identifier columns from:
1. Explicit `identifiers` vector
2. `ColumnHint(kind=:identifier)` entries
3. Auto-detection (string/integer columns with ≥90% distinct non-missing values)
"""
function _resolve_identifiers(col_names::Vector{Symbol},
                               identifiers::Vector{Symbol},
                               hints::Vector{ColumnHint},
                               nm_cache::Dict{Symbol, Vector},
                               basetype_cache::Dict{Symbol, Type})
    id_set = Set{Symbol}(identifiers)

    # Add hint-based identifiers
    for h in hints
        h.kind == :identifier && push!(id_set, h.name)
    end

    # Columns with any explicit hint should not be auto-detected
    hinted = Set(h.name for h in hints)

    # Auto-detect: string or integer columns with ≥90% distinct non-missing
    for name in col_names
        name in id_set && continue
        name in hinted && continue

        nm = nm_cache[name]
        isempty(nm) && continue

        T = basetype_cache[name]
        (T <: AbstractString || T <: Integer) || continue

        n_nonmissing = length(nm)
        n_unique = length(unique(nm))
        ratio = n_unique / n_nonmissing

        # Uniqueness among a handful of observations is not evidence of a key.
        # A column with a single non-missing value scores a ratio of 1.0 and is
        # obviously a constant; without this floor it was classified as an
        # identifier and silently dropped from the output. Found on OpenML's
        # `anneal`, whose :bc column has 898 rows and one observation.
        if n_nonmissing < MIN_IDENTIFIER_OBSERVATIONS
            continue
        end

        if ratio >= 0.9
            @info "Column :$name auto-detected as identifier " *
                  "(N_unique/N_nonmissing = $(round(ratio, digits=2))); " *
                  "excluding from model. Pass hints to override."
            push!(id_set, name)
        end
    end

    return id_set
end

# ─── Fill spec application ──────────────────────────────────────────────────

"""
    _apply_fill(spec, col_name::Symbol, n::Int) -> Vector

Generate `n` fill values for an identifier column.

Fill specs:
- `:sequential`     → `"<colname>_1"`, `"<colname>_2"`, ...
- `:sequential_int` → `1`, `2`, `3`, ...
- `"prefix"`        → `"prefix_1"`, `"prefix_2"`, ...
- `f::Function`     → `f(1)`, `f(2)`, ...
"""
function _apply_fill(spec::Symbol, col_name::Symbol, n::Int)
    if spec === :sequential
        return ["$(col_name)_$i" for i in 1:n]
    elseif spec === :sequential_int
        return collect(1:n)
    else
        throw(ArgumentError(
            "Unknown fill Symbol :$spec for :$col_name; " *
            "expected :sequential or :sequential_int"))
    end
end

function _apply_fill(spec::AbstractString, ::Symbol, n::Int)
    return ["$(spec)_$i" for i in 1:n]
end

function _apply_fill(spec::Function, ::Symbol, n::Int)
    return [spec(i) for i in 1:n]
end

function _apply_fill(spec, col_name::Symbol, ::Int)
    throw(ArgumentError(
        "Invalid fill spec for :$col_name: expected :sequential, " *
        ":sequential_int, a String prefix, or a Function; " *
        "got $(typeof(spec))"))
end
