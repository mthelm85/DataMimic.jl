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
