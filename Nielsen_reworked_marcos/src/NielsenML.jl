module NielsenML

using Random, DelimitedFiles, LinearAlgebra

include("network.jl")
include("train.jl")
include("minst.jl")
include("sigmoid_cost.jl")

end # module
