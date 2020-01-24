# UNet.jl

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

To default input channel dimension is expected to be `1` ie. grayscale. To support different channel images, you can pass the `channels` to `Unet`.

```julia
julia> u = Unet(3) # for RGB images
```

The input size can be any power of two sized batch. Something like `(256,256, channels, batch_size)`.

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

## Further Reading
The package is an implementation of the [paper](https://arxiv.org/pdf/1505.04597.pdf), and all credits of the model itself go to the respective authors.
