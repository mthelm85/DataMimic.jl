using Documenter
using DataMimic
using MaterialDocs

makedocs(
    sitename = "DataMimic.jl",
    modules  = [DataMimic],
    authors  = "Matt Helm",
    format   = Material3(;
        logo = "assets/logo.png",
        favicon = "assets/favicon.png",
        # Directory-style URLs (./engines/) can't be followed when the built
        # docs are opened from disk, so keep them for CI only.
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
