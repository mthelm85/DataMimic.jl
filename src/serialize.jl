# ─── Serialization ───────────────────────────────────────────────────────────

const DATAMIMIC_SERIAL_VERSION = v"2.0.0"

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

Deserialize a fitted model from disk. Throws if the file was saved
with an incompatible DataMimic version.
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
            error("Model was saved with DataMimic v$ver, which is " *
                  "incompatible with the current version " *
                  "v$DATAMIMIC_SERIAL_VERSION. " *
                  "Major version mismatch; the model must be re-fitted.")
        end
        if ver > DATAMIMIC_SERIAL_VERSION
            @warn "Model was saved with DataMimic v$ver, which is " *
                  "newer than the current v$DATAMIMIC_SERIAL_VERSION. " *
                  "Loading anyway, but some features may be missing."
        end
        Serialization.deserialize(io)
    end
end
