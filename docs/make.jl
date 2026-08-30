using Documenter
using DataMimic

makedocs(
    sitename = "DataMimic.jl",
    modules  = [DataMimic],
    authors  = "Matt Helm",
    format   = Documenter.HTML(
        canonical = "https://mthelm85.github.io/DataMimic.jl",
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home"        => "index.md",
        "Engines"     => "engines.md",
        "Privacy"     => "privacy.md",
        "Evaluation"  => "evaluation.md",
        "API"         => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo   = "github.com/mthelm85/DataMimic.jl",
    devbranch = "main",
)
