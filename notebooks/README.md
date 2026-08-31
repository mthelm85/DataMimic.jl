# Notebooks

## `guided-tour.qmd`

A deep dive into DataMimic: how to use it, and how each engine works. Four
real datasets — Adult, German Credit, Wine Quality, Covertype — chosen so
that a different engine wins on each, with the methods opened up far enough
to explain why.

```bash
quarto render notebooks/guided-tour.qmd
```

Needs [Quarto](https://quarto.org) 1.5 or later, which has a native Julia
engine. The environment here is separate from the package's own, so
dependencies (CairoMakie in particular) do not enter the test or benchmark
environments:

```bash
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

Datasets download on first run (about 60 MB, mostly Covertype) and cache under
`benchmark/data/`, shared with the benchmark suite.

The first render takes roughly half an hour. Almost all of that is two
`DiffusionGenerator` models, trained at two budgets so the document can show
what an undertrained diffusion model looks like and how it improves — the rest
of the tour runs in seconds.

Chunk results are cached under `notebooks/.jupyter_cache`, so re-rendering
after a prose edit is fast. Delete that directory to force a clean run.

## Status

This is a first draft. It executes end to end and its numbers are real, but it
has not been through an editing pass and is not linked from the README or the
documentation site yet.
