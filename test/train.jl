using Test, UNet
using UNet.Flux, UNet.Flux.Zygote
using Base.Iterators
using FileIO
using Images

# test_dir_input ="/home/rzietal/git/UNet.jl/test/data/png/testing/input/"
# test_dir_target ="/home/rzietal/git/UNet.jl/test/data/png/testing/target/"

# train_dir_input ="/home/rzietal/git/UNet.jl/test/data/png/training/input/"
# train_dir_target ="/home/rzietal/git/UNet.jl/test/data/png/training/target/"

# model_dir = "/home/rzietal/git/UNet.jl/test/data/models"

test_dir_input ="D:\\Projects\\UNet.jl\\test\\data\\png\\overfit\\input"
test_dir_target ="D:\\Projects\\UNet.jl\\test\\data\\png\\overfit\\target"

train_dir_input ="D:\\Projects\\UNet.jl\\test\\data\\png\\training\\input"
train_dir_target ="D:\\Projects\\UNet.jl\\test\\data\\png\\training\\target"

model_dir = "D:\\Projects\\UNet.jl\\test\\data\\models"

nepochs = 1000
numfiles = 30
batchsize = 5
lr = 0.01
lr_drop_rate = 0.95
lr_step = 50
train(train_dir_input, train_dir_target, test_dir_input, test_dir_target, nepochs, numfiles, batchsize, lr, lr_drop_rate, lr_step, model_dir)
