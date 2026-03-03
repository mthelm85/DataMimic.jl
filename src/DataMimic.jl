module DataMimic

include("types.jl")
include("detect.jl")
include("fit.jl")
include("sample.jl")

export SynthModel
export fit, sample, synthesize

end
