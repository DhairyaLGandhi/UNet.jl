function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2),
	  BatchNorm(out_ch),
	  x->squeeze(x))
end

UNetConvBlock(in_chs, out_chs, kernel = (3, 3)) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=_random_normal),
	BatchNormWrap(out_chs)...,
	x->leakyrelu.(x,0.2f0))

ConvDown(in_chs,out_chs,kernel = (4,4)) =
  Chain(Conv(kernel,in_chs=>out_chs,pad=(1,1),stride=(2,2);init=_random_normal),
	BatchNormWrap(out_chs)...,
	x->leakyrelu.(x,0.2f0))

struct UNetUpBlock
  upsample
end

@functor UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), p = 0.5f0) = 
    UNetUpBlock(Chain(x->leakyrelu.(x,0.2f0),
       		ConvTranspose((2, 2), in_chs=>out_chs,
			stride=(2, 2);init=_random_normal),
		BatchNormWrap(out_chs)...,
		Dropout(p)))

function (u::UNetUpBlock)(x, bridge)
  x = u.upsample(x)
  return cat(x, bridge, dims = 3)
end

"""
    Unet(channels::Int = 1)

  Initializes a [UNet](https://arxiv.org/pdf/1505.04597.pdf) instance with the given number of channels, typically equal to the number of channels in the input images.
"""
struct Unet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor Unet

function Unet(channels::Int = 1)
  conv_down_blocks = Chain(ConvDown(64,64),
		      ConvDown(128,128),
		      ConvDown(256,256),
		      ConvDown(512,512))

  conv_blocks = Chain(UNetConvBlock(channels, 3),
		 UNetConvBlock(3, 64),
		 UNetConvBlock(64, 128),
		 UNetConvBlock(128, 256),
		 UNetConvBlock(256, 512),
		 UNetConvBlock(512, 1024),
		 UNetConvBlock(1024, 1024))

  up_blocks = Chain(UNetUpBlock(1024, 512),
		UNetUpBlock(1024, 256),
		UNetUpBlock(512, 128),
		UNetUpBlock(256, 64,p = 0.0f0),
		Chain(x->leakyrelu.(x,0.2f0),
		Conv((1, 1), 128=>channels;init=_random_normal)))									  
  Unet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::Unet)(x)
  outputs = Vector(undef, 5)
  outputs[1] = u.conv_blocks[1:2](x)

  for i in 2:5
    pool_x = u.conv_down_blocks[i - 1](outputs[i - 1])
    outputs[i] = u.conv_blocks[i+1](pool_x)
  end

  up_x = u.conv_blocks[7](outputs[end])

  for i in 1:4
    up_x = u.up_blocks[i](up_x, outputs[end - i])
  end

  tanh.(u.up_blocks[end](up_x))
end

function Base.show(io::IO, u::Unet)
  println(io, "UNet:")

  for l in u.conv_down_blocks
    println(io, "  ConvDown($(size(l[1].weight)[end-1]), $(size(l[1].weight)[end]))")
  end

  println(io, "\n")
  for l in u.conv_blocks
    println(io, "  UNetConvBlock($(size(l[1].weight)[end-1]), $(size(l[1].weight)[end]))")
  end

  println(io, "\n")
  for l in u.up_blocks
    l isa UNetUpBlock || continue
    println(io, "  UNetUpBlock($(size(l.upsample[2].weight)[end]), $(size(l.upsample[2].weight)[end-1]))")
  end
end
