# ─── Serialization ───────────────────────────────────────────────────────────

# Version of the on-disk model format, deliberately independent of the
# package version: `load` rejects a mismatched major, and under 0.x semver the
# breaking axis is the minor, so tracking the package version here would stop
# this catching breaking format changes. Bump the major whenever a change makes
# previously-saved models unreadable.
const DATAMIMIC_SERIAL_VERSION = v"1.0.0"

"""
    DataMimic.save(path::AbstractString, model::AbstractFittedModel)

Serialize a fitted model to disk with a version header.

**Note:** if a `fill` spec contains a `Function`, it is serialized via
Julia's `Serialization` module. Anonymous functions round-trip correctly
within the same Julia version but may fail across versions.
"""
function save(path::AbstractString, model::AbstractFittedModel)
    open(path, "w") do io
        Serialization.serialize(io, DATAMIMIC_SERIAL_VERSION)
        Serialization.serialize(io, model)
    end
    return path
end

"""
    DataMimic.load(path::AbstractString) -> AbstractFittedModel

Deserialize a fitted model from disk. Throws if the file was saved in an
incompatible model format. The format version is independent of the package
version; see `DATAMIMIC_SERIAL_VERSION`.
"""
function load(path::AbstractString)
    return open(path, "r") do io
        ver = Serialization.deserialize(io)
        if !(ver isa VersionNumber)
            error("File does not appear to be a DataMimic model " *
                  "(missing version header). " *
                  "Was it saved with an older version?")
        end
        if ver.major != DATAMIMIC_SERIAL_VERSION.major
            error("Model was saved in format v$ver, which is incompatible " *
                  "with the format this version of DataMimic reads " *
                  "(v$DATAMIMIC_SERIAL_VERSION). Major version mismatch; " *
                  "the model must be re-fitted.")
        end
        if ver > DATAMIMIC_SERIAL_VERSION
            @warn "Model was saved in format v$ver, which is newer than " *
                  "the format this version of DataMimic reads " *
                  "(v$DATAMIMIC_SERIAL_VERSION). Loading anyway, but some " *
                  "features may be missing."
        end
        Serialization.deserialize(io)
    end
end
