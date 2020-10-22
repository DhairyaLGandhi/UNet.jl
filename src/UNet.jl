module UNet

using StatsBase
using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux: logitcrossentropy

using ImageCore
using ImageTransformations: imresize
using FileIO
#using Distributions: Normal
using Serialization
using ForwardDiff
using Parameters: @with_kw
using CUDAapi
using CUDA

include("defaults.jl")
include("utils.jl")
include("dataloader.jl")
include("model.jl")
include("train.jl")

export Unet, train

end # module
