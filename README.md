# UNet.jl

[![Actions Status](https://github.com/dhairyagandhi96/UNet.jl/workflows/CI/badge.svg)](https://github.com/dhairyagandhi96/UNet.jl/actions)

This pacakge provides a generic UNet implemented in Julia.

The package is built on top of Flux.jl, and therefore can be extended as needed

```julia
julia> u = Unet()
UNet:
  ConvDown(64, 64)
  ConvDown(128, 128)
  ConvDown(256, 256)
  ConvDown(512, 512)


  UNetConvBlock(1, 3)
  UNetConvBlock(3, 64)
  UNetConvBlock(64, 128)
  UNetConvBlock(128, 256)
  UNetConvBlock(256, 512)
  UNetConvBlock(512, 1024)
  UNetConvBlock(1024, 1024)


  UNetUpBlock(1024, 512)
  UNetUpBlock(1024, 256)
  UNetUpBlock(512, 128)
  UNetUpBlock(256, 64)
```

The default input channel dimension is expected to be `1` ie. grayscale. To support different channel images, you can pass the `channels` to `Unet`.

```julia
julia> u = Unet(3) # for RGB images
```

The input size can be any power of two sized batch. Something like `(256,256, channels, batch_size)`.

The default output channel dimension is the input channel dimension. So, `1` for a `Unet()` and e.g. `3` for a `Unet(3)`.
The output channel dimension can be set by supplying a second argument:

```julia
julia> u = Unet(3, 5) # 3 input channels, 5 output channels.
```

## GPU Support

To train the model on UNet, it is as simple as calling `gpu` on the model.

```julia
julia> u = gpu(u);

julia> r = gpu(rand(Float32, 256, 256, 1, 1));

julia> size(u(r))
(256, 256, 1, 1)
```

## Training

Training UNet is a breeze too.

You can define your own loss function, or use a provided Binary Cross Entropy implementation via `bce`.

```julia
julia> w = rand(Float32, 256,256,1,1);

julia> w′ = rand(Float32, 256,256,1,1);

julia> function loss(x, y)
         op = clamp.(u(x), 0.001f0, 1.f0)
         mean(bce(op, y))
       end
loss (generic function with 1 method)

julia> using Base.Iterators

julia> rep = Iterators.repeated((w, w′), 10);

julia> opt = Momentum()
Momentum(0.01, 0.9, IdDict{Any,Any}())

julia> Flux.train!(loss, Flux.params(u), rep, opt);
```

## Further Reading
The package is an implementation of the [paper](https://arxiv.org/pdf/1505.04597.pdf), and all credits of the model itself go to the respective authors.
