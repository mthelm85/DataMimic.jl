# ─── Dataset loaders for DataMimic benchmarks ─────────────────────────────
#
# Each function downloads (if needed), caches, and returns a cleaned DataFrame
# ready for use with DataMimic.fit().

using CSV, DataFrames, Downloads, CodecZlib

const DATA_DIR = joinpath(@__DIR__, "data")

function ensure_data_dir()
    isdir(DATA_DIR) || mkpath(DATA_DIR)
end

# ─── Adult (Census Income) ─────────────────────────────────────────────────

const ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
const ADULT_TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

const ADULT_COLUMNS = [
    :age, :workclass, :fnlwgt, :education, :education_num,
    :marital_status, :occupation, :relationship, :race, :sex,
    :capital_gain, :capital_loss, :hours_per_week, :native_country,
    :income
]

"""
    load_adult(; combine=true) -> DataFrame

Load the UCI Adult (Census Income) dataset.
- 14 features (6 continuous, 8 categorical) + 1 binary target (:income)
- ~48,842 rows when `combine=true` (train + test)
"""
# ─── Adult value normalisation ─────────────────────────────────────────────
#
# The raw files pad every field with a leading space (" State-gov"), and mark
# missing entries with "?". Passing `missingstring = " ?"` to CSV.read does not
# catch them, so for a long time this loader returned 6,465 entries across
# :workclass, :occupation and :native_country as the literal string " ?" — a
# category level rather than a missing value. UCI documents "?" as the missing
# marker, so treat it as one, and strip the padding while we are here.
function _normalize_adult!(df::DataFrame)
    for c in names(df)
        col = df[!, c]
        eltype(col) <: Union{Missing, AbstractString} || continue
        cleaned = Vector{Union{Missing, String}}(undef, length(col))
        for (i, v) in enumerate(col)
            if ismissing(v)
                cleaned[i] = missing
            else
                t = strip(String(v))
                cleaned[i] = t == "?" ? missing : t
            end
        end
        df[!, c] = cleaned
    end
    return df
end

function load_adult(; combine::Bool = true)
    ensure_data_dir()

    train_path = joinpath(DATA_DIR, "adult_train.csv")
    test_path  = joinpath(DATA_DIR, "adult_test.csv")

    if !isfile(train_path)
        @info "Downloading Adult training data..."
        Downloads.download(ADULT_TRAIN_URL, train_path)
    end
    if !isfile(test_path)
        @info "Downloading Adult test data..."
        Downloads.download(ADULT_TEST_URL, test_path)
    end

    # Parse — no header, whitespace after commas, missing = " ?"
    train = CSV.read(train_path, DataFrame;
                     header = false,
                     missingstring = " ?",
                     stripwhitespace = true)

    # Test file has a junk first line "|1x3 Cross validator"
    test = CSV.read(test_path, DataFrame;
                    header = false,
                    missingstring = " ?",
                    stripwhitespace = true,
                    skipto = 2)

    rename!(train, [ADULT_COLUMNS...])
    rename!(test,  [ADULT_COLUMNS...])

    # Clean up target: test set has trailing "." on labels
    test.income = replace.(test.income, r"\.$" => "")

    _normalize_adult!(train)
    _normalize_adult!(test)

    df = combine ? vcat(train, test) : train

    # Drop fnlwgt (sampling weight, not a real feature)
    select!(df, Not(:fnlwgt))

    @info "Adult dataset loaded" rows=nrow(df) cols=ncol(df)
    return df
end

# ─── Covertype (Forest Cover Type) ────────────────────────────────────────

const COVERTYPE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

"""
    load_covertype(; n=nothing) -> DataFrame

Load the UCI Covertype dataset.
- 54 features (10 continuous, 44 binary) + 1 target (:cover_type)
- 581,012 rows (subsample with `n` for faster benchmarks)
"""
function load_covertype(; n::Union{Nothing, Int} = nothing)
    ensure_data_dir()

    gz_path  = joinpath(DATA_DIR, "covtype.data.gz")
    csv_path = joinpath(DATA_DIR, "covtype.csv")

    if !isfile(csv_path)
        if !isfile(gz_path)
            @info "Downloading Covertype data..."
            Downloads.download(COVERTYPE_URL, gz_path)
        end
        @info "Decompressing Covertype data..."
        compressed = read(gz_path)
        decompressed = transcode(GzipDecompressor, compressed)
        write(csv_path, decompressed)
    end

    col_names = vcat(
        [:elevation, :aspect, :slope,
         :horizontal_distance_hydrology, :vertical_distance_hydrology,
         :horizontal_distance_roadways, :hillshade_9am, :hillshade_noon,
         :hillshade_3pm, :horizontal_distance_firepoints],
        [Symbol("wilderness_area_$i") for i in 1:4],
        [Symbol("soil_type_$i") for i in 1:40],
        [:cover_type]
    )

    df = CSV.read(csv_path, DataFrame; header = false)
    rename!(df, col_names)

    # Subsample if requested
    if n !== nothing && n < nrow(df)
        df = df[shuffle(1:nrow(df))[1:n], :]
    end

    @info "Covertype dataset loaded" rows=nrow(df) cols=ncol(df)
    return df
end

# ─── German Credit (Statlog) ───────────────────────────────────────────────

const GERMAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# The raw file codes every categorical level as "A" plus a number, which is
# unreadable in a synthetic sample. These maps restore the documented meanings
# so the output can be eyeballed.
const GERMAN_LEVELS = Dict(
    :checking_status  => Dict("A11" => "<0DM", "A12" => "0-200DM",
                              "A13" => ">=200DM", "A14" => "none"),
    :credit_history   => Dict("A30" => "none_taken", "A31" => "all_paid",
                              "A32" => "existing_paid", "A33" => "delayed",
                              "A34" => "critical"),
    :purpose          => Dict("A40" => "car_new", "A41" => "car_used",
                              "A42" => "furniture", "A43" => "radio_tv",
                              "A44" => "appliances", "A45" => "repairs",
                              "A46" => "education", "A47" => "vacation",
                              "A48" => "retraining", "A49" => "business",
                              "A410" => "other"),
    :savings          => Dict("A61" => "<100DM", "A62" => "100-500DM",
                              "A63" => "500-1000DM", "A64" => ">=1000DM",
                              "A65" => "unknown"),
    :employment_since => Dict("A71" => "unemployed", "A72" => "<1yr",
                              "A73" => "1-4yr", "A74" => "4-7yr", "A75" => ">=7yr"),
    :personal_status  => Dict("A91" => "male_divorced", "A92" => "female_div_sep_mar",
                              "A93" => "male_single", "A94" => "male_married",
                              "A95" => "female_single"),
    :other_debtors    => Dict("A101" => "none", "A102" => "co_applicant",
                              "A103" => "guarantor"),
    :property         => Dict("A121" => "real_estate", "A122" => "savings_agreement",
                              "A123" => "car", "A124" => "unknown"),
    :other_plans      => Dict("A141" => "bank", "A142" => "stores", "A143" => "none"),
    :housing          => Dict("A151" => "rent", "A152" => "own", "A153" => "free"),
    :job              => Dict("A171" => "unemployed_nonres", "A172" => "unskilled_res",
                              "A173" => "skilled", "A174" => "management"),
    :telephone        => Dict("A191" => "none", "A192" => "yes"),
    :foreign_worker   => Dict("A201" => "yes", "A202" => "no"),
)

"""
    load_german_credit() -> DataFrame

UCI Statlog (German Credit). 1,000 rows, 13 categorical and 7 numeric
attributes, with a binary `:credit_risk` target.

Small and categorical-heavy, which is the shape `MSTGenerator` is built for —
and a credit file is the kind of data where a formal privacy guarantee is the
point rather than a nicety.
"""
function load_german_credit()
    ensure_data_dir()
    path = joinpath(DATA_DIR, "german.data")
    if !isfile(path)
        @info "Downloading German Credit data..."
        Downloads.download(GERMAN_URL, path)
    end

    names_ = [:checking_status, :duration_months, :credit_history, :purpose,
              :credit_amount, :savings, :employment_since, :installment_rate,
              :personal_status, :other_debtors, :residence_since, :property,
              :age, :other_plans, :housing, :existing_credits, :job,
              :dependents, :telephone, :foreign_worker, :credit_risk]

    df = CSV.read(path, DataFrame; header = false, delim = ' ', ignorerepeated = true)
    rename!(df, names_)

    for (col, map) in GERMAN_LEVELS
        df[!, col] = [get(map, v, v) for v in df[!, col]]
    end
    df.credit_risk = [v == 1 ? "good" : "bad" for v in df.credit_risk]

    return df
end

# ─── Wine Quality ──────────────────────────────────────────────────────────

const WINE_WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
const WINE_RED_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

"""
    load_wine(; colour = :white) -> DataFrame

UCI Wine Quality. 4,898 white (or 1,599 red) rows, 11 continuous physico-
chemical measurements plus an integer `:quality` score.

Entirely continuous, and strongly correlated — density against residual sugar
and alcohol especially — which makes it the clearest place to watch a copula
reproduce a dependence structure, and the fairest ground for
`DPCopulaGenerator`.
"""
function load_wine(; colour::Symbol = :white)
    colour in (:white, :red) || throw(ArgumentError("colour must be :white or :red"))
    ensure_data_dir()
    fname = colour === :white ? "winequality-white.csv" : "winequality-red.csv"
    url   = colour === :white ? WINE_WHITE_URL : WINE_RED_URL
    path  = joinpath(DATA_DIR, fname)
    if !isfile(path)
        @info "Downloading Wine Quality ($colour)..."
        Downloads.download(url, path)
    end

    df = CSV.read(path, DataFrame; delim = ';')
    rename!(df, Dict(n => Symbol(replace(string(n), " " => "_")) for n in names(df)))
    return df
end

# ─── OpenML ARFF datasets ─────────────────────────────────────────────────
#
# A small registry rather than an API client: the download URLs are resolved
# once and pinned, so a benchmark run does not depend on OpenML's metadata
# service being up or on adding a JSON dependency to this environment.
#
# These carry the property the UCI datasets above lack — genuinely
# high-cardinality categorical columns — which is what MST's domain
# compression acts on.

const OPENML_DATASETS = Dict(
    4552  => (name = "BachChoralHarmony",
              url  = "https://www.openml.org/data/v1/download/1798821/BachChoralHarmony.arff",
              note = "5,665x17, nominal-dominated, 102- and 60-level columns"),
    41160 => (name = "rl",
              url  = "https://www.openml.org/data/v1/download/19335533/rl.arff",
              note = "31,406x23, mixed, one 1,855-level column"),
    473   => (name = "cjs",
              url  = "https://www.openml.org/data/v1/download/52585/cjs.arff",
              note = "2,796x35, numeric-dominated, 3 nominal columns"),
    516   => (name = "pbcseq",
              url  = "https://www.openml.org/data/v1/download/52628/pbcseq.arff",
              note = "1,945x19, small n, numeric-heavy"),
)

"""
    load_openml(id::Int) -> DataFrame

Download (once), cache, and parse an OpenML ARFF dataset from `OPENML_DATASETS`.

Column types come from the `@attribute` declarations rather than from guessing
at the values: several nominal columns in these tables hold levels that parse
cleanly as numbers and must not be treated as numeric. Missing values (`?`)
become `missing`.

Rows whose field count disagrees with the header are dropped, so a parser
problem stays distinguishable from a data problem.
"""
function load_openml(id::Int)
    haskey(OPENML_DATASETS, id) ||
        throw(ArgumentError("unknown OpenML id $id; known: " *
                            join(sort(collect(keys(OPENML_DATASETS))), ", ")))
    spec = OPENML_DATASETS[id]
    ensure_data_dir()
    path = joinpath(DATA_DIR, "openml_$(id)_$(spec.name).arff")
    if !isfile(path)
        @info "Downloading OpenML $id ($(spec.name))..."
        Downloads.download(spec.url, path)
    end

    numeric = Bool[]
    rows    = Vector{Vector{Union{Missing,String}}}()
    in_data = false
    trim    = ['\'', '"', ' ']

    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "%")) && continue
        low = lowercase(line)
        if !in_data && startswith(low, "@attribute")
            m = match(r"^@attribute\s+(\"[^\"]*\"|'[^']*'|\S+)\s+(.*)$"i, line)
            m === nothing && continue
            push!(numeric, occursin(r"^(numeric|real|integer)"i,
                                    strip(String(m.captures[2]))))
        elseif !in_data && startswith(low, "@data")
            in_data = true
        elseif in_data
            fields = map(split(line, ',')) do f
                t = strip(f, trim)
                (t == "?" || isempty(t)) ? missing : String(t)
            end
            push!(rows, collect(fields))
        end
    end

    isempty(numeric) && error("no @attribute lines found in $path")
    width = length(numeric)
    rows  = filter(r -> length(r) == width, rows)
    isempty(rows) && error("no well-formed data rows in $path")

    df = DataFrame()
    for j in 1:width
        col = [r[j] for r in rows]
        if numeric[j]
            parsed = map(v -> ismissing(v) ? missing :
                              something(tryparse(Float64, v), missing), col)
            df[!, Symbol("V", j)] = any(ismissing, parsed) ?
                Vector{Union{Missing,Float64}}(parsed) : Vector{Float64}(parsed)
        else
            df[!, Symbol("V", j)] = any(ismissing, col) ?
                Vector{Union{Missing,String}}(col) : Vector{String}(col)
        end
    end
    return df
end
