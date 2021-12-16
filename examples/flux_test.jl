# Test FLUX.jl and a U-net architechture

using Flux
using UNet
using TestImages
using View5D
using Noise
using NDTools
using FourierTools
using IndexFunArrays

img = 10.0 .* Float32.(testimage("resolution_test_512"))


# @ve nimg1 nimg2

u = Unet(1, 1, 64, 2, [(2,2),(2,2),(2,2),(2,2)], [[(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)]], 
[[(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)]], NNlib.relu, NNlib.relu; padding="same"); 
u = gpu(u);
function loss(x, y)
    # op = clamp.(u(x), 0.001f0, 1.f0)
    mean(abs2.(u(x) .-y))
end
opt = Momentum()

function get_random_tile(img, tile_size=(128,128), ctr = (rand(tile_size[1]÷2:size(img,1)-tile_size[1]÷2),rand(tile_size[2]÷2:size(img,2)-tile_size[2]÷2)) )
    return select_region(img,new_size=tile_size, center=ctr), ctr
end

sz = size(img)
psf = abs2.(ift(disc(sz, 40))); psf ./= sum(psf)
conv_img = conv_psf(img,psf)

scale = 0.5/maximum(conv_img)
patch = (128,128)
for n in 1:100
    println("Iteration: $n")
    myimg, pos = get_random_tile(conv_img,patch)
    nimg1 = gpu(scale.*reshape(poisson(myimg),(size(myimg)...,1,1)))
    # nimg2 = gpu(scale.*reshape(poisson(myimg),(size(myimg)...,1,1)))
    pimg, pos = get_random_tile(img,patch,pos)
    pimg = gpu(scale.*reshape(pimg,(size(myimg)...,1,1)))
    rep = Iterators.repeated((nimg1, pimg), 1);
    Flux.train!(loss, Flux.params(u), rep, opt)
end

# myimg, pos = get_random_tile(conv_img,patch)
# nimg3 = gpu(scale.*reshape(poisson(myimg),(size(myimg)...,1,1)))
# @ve myimg nimg3 u(nimg3)

# apply the net to the whole image instead:
nimg = gpu(scale.*reshape(poisson(conv_img),(size(conv_img)...,1,1)))
@ve img nimg u(nimg)
