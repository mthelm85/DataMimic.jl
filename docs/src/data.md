# Preparing your data

Most tables need no preparation: pass them to `fit` and the column types are
worked out for you. This page covers the cases where that inference needs
help, and what happens to identifiers and missing values along the way.

## Column type detection

Every column is classified as one of:

| Kind | Meaning |
|---|---|
| `:continuous` | Real-valued |
| `:integer` | Whole numbers with enough distinct values to model as numeric |
| `:categorical` | A modest set of unordered levels |
| `:binary` | Two levels, or a `Bool` column |
| `:constant` | A single value throughout |
| `:identifier` | Effectively unique per row — a key, not a variable |

Detection is cardinality-aware, which matters most for integers. A column of
whole numbers holding five distinct values in ten thousand rows is a category
that happens to be numbered, and modelling it as continuous would invent values
between the levels. The same column with ten thousand distinct values is a
genuine numeric variable. Detection draws that line by the ratio of distinct
values to rows, not by the element type.

The classification is reported when it may surprise you — an auto-detected
identifier logs an informational message naming the column and the ratio that
triggered it.

### Dates and times

`Date`, `DateTime` and `Time` columns are `:continuous`. They are modelled on
their own numeric scale — days for `Date`, milliseconds for `DateTime` — and
come back as the same type they went in as.

Two things follow from that. Chronology is preserved, because the model sees
an ordered quantity rather than a set of unrelated labels; and synthetic dates
can fall *between* the observed ones, rather than only reusing dates that
appear in the input.

It also keeps the model small. Treating dates as categories gives every
distinct timestamp its own level, and dates are usually near-unique. On a
1,400-row personnel table, eight date columns expanded to 2,447 of 5,843 model
dimensions that way — more dimensions than rows, which is enough to stop
`DiffusionGenerator` converging at all.

### Overriding with a hint

When the inference gets it wrong, say so with a [`ColumnHint`](@ref):

```julia
model = fit(CopulaGenerator(), df;
            hints = [ColumnHint(name = :zip_code, kind = :categorical)])
```

Postal codes, year columns, and coded survey responses are the usual cases:
numeric in storage, categorical in meaning. A hint may also supply the full
level set for a categorical column, which is useful when your sample does not
happen to contain every level that exists:

```julia
ColumnHint(name = :grade, kind = :categorical, levels = ["A", "B", "C", "D", "F"])
```

Observed values not covered by an explicit `levels` list are excluded from the
marginal, and you get a warning listing them.

## Identifiers

An identifier column carries no statistical signal — one value per row, by
definition — but it does carry disclosure risk, so DataMimic keeps it out of
the model entirely. Real identifier values never reach the synthetic table.

Because the column is not modelled, it cannot be sampled. **Without a `fill`
spec it is dropped from the output.** Give it one to regenerate the column:

```julia
df = DataFrame(
    ein     = ["12-3456789", "98-7654321", "55-1122334"],
    amount  = [1200.0, 850.0, 2310.0],
    quarter = ["Q1", "Q2", "Q1"],
)

model = fit(CopulaGenerator(), df;
            identifiers = [:ein],
            fill        = Dict(:ein => :sequential))

syn = sample(model, 100)     # ein = "ein_1", "ein_2", …
```

A fill spec is one of:

| Spec | Result |
|---|---|
| `:sequential` | `"<colname>_1"`, `"<colname>_2"`, … |
| `:sequential_int` | `1`, `2`, `3`, … |
| a `String` prefix | `"prefix_1"`, `"prefix_2"`, … |
| a `Function` | `f(i)` for row `i` |

The function form covers formats that need to look real:

```julia
fill = Dict(:ein => i -> string(lpad(10_000_000 + i, 8, '0')[1:2], "-",
                               lpad(10_000_000 + i, 8, '0')[3:end]))
```

Identifiers are excluded rather than obfuscated deliberately. Shuffling the
characters of a real identifier leaves a value derived from a real record, and
the shuffle is often reversible; a freshly generated value is not derived from
anyone.

## Missing values

Missingness is measured per column at fit time and reintroduced at the same
rate when sampling, so the synthetic table has a comparable missingness
profile rather than a suspiciously complete one.

The rate is modelled, not the pattern: whether a value is missing is drawn
independently per cell, so if missingness in your data is correlated across
columns — a block of questions skipped together, say — that structure is not
reproduced. A column that is entirely missing is treated as constant.

## Reproducibility

Every stochastic entry point takes an `rng`:

```julia
model = fit(CopulaGenerator(), df; rng = MersenneTwister(42))
syn   = sample(model, 500)                       # uses the model's rng
syn2  = sample(model, 500; rng = MersenneTwister(7))   # or pass another
```

The `rng` given to `fit` is stored on the model, so `sample` is reproducible
without repeating yourself. Fitting twice from equal seeds gives identical
models.

Note that this fixes *your* randomness, not the seed-to-seed variance of the
method itself. Two different seeds can produce meaningfully different synthetic
tables, particularly at tight privacy budgets — which is why
[`compare`](@ref) repeats each engine over several seeds and reports the spread.
