module UNet

using StatsBase
using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux: logitcrossentropy, dice_coeff_loss

using Images
using ImageCore
using ImageTransformations: imresize
using FileIO

using Serialization
using ForwardDiff
using Parameters: @with_kw
using CUDAapi
using CUDA

include("defaults.jl")
include("dataloader.jl")
include("model.jl")
include("train.jl")

export Unet, train

end # module
