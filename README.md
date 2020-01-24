# UNet.jl

This pacakge provides a generic UNet implemented in Julia.

The package is built on top of Flux.jl, and therefore can be extended as needed

```julia
julia> u = Unet()
```

To default input channel dimension is expected to be `1` ie. grayscale. To support different channel images, you can pass the `channels` to `Unet`.

```julia
julia> u = Unet(3) # for RGB images
```

## GPU Support

To train the model on UNet, it is as simple as calling `gpu` on the model.

```julia
julia> gpu(u)
```
