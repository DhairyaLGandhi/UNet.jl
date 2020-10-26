DoubleConv(in_channels, out_channels; conv_kernel = (3,3), conv_pad = (1,1)) = 
    Chain(
        Conv(conv_kernel, in_channels=>out_channels, pad = conv_pad),
        BatchNorm(out_channels, relu),
        Conv(conv_kernel, out_channels=>out_channels, pad = conv_pad),
        BatchNorm(out_channels, relu),
    )

PoolDrop(;pool_kernel = (2,2), pool_stride = (2,2), pool_padding = (1,1), drop_prob = 0.5) = 
    Chain(
        MaxPool(pool_kernel; pad = pool_padding, stride = pool_stride),
        Dropout(drop_prob)
    )

struct UpSample
    upsample
end

@functor UpSample

UpSample(in_channels, out_channels; dconv_kernel = (3,3), dconv_pad = (1,1), dconv_stride = (2,2), drop_prob = 0.5) = 
    UpSample(
        Chain(
            ConvTranspose(dconv_kernel, in_channels=>out_channels, stride=dconv_stride, pad = dconv_pad), 
            Dropout(drop_prob)   
        )
    )

function (u::UpSample)(x::AbstractArray{T}, bridge::AbstractArray{T}) where T
    x = u.upsample(x)
    s1 = size(x)
    s2 = size(bridge)
    s = min(s1[1],s2[1])
    #possibly match of dimensions needed here
    return cat(x[1:s,1:s,:,:], bridge[1:s,1:s,:,:], dims = 3)
end


struct Unet
    double_conv_down_1
    double_conv_down_2
    double_conv_down_3
    double_conv_down_4
    double_conv_down_5
    double_conv_up_1
    double_conv_up_2
    double_conv_up_3
    double_conv_up_4
    upsample_1
    upsample_2
    upsample_3
    upsample_4
    pooldrop
    activation
end

@functor Unet

function Unet(channels::Int = 3, nlabels::Int = 7, nfilters::Int = 16)

    Unet(
        DoubleConv(channels,nfilters), # double_conv_down_1
        DoubleConv(nfilters,2*nfilters), # double_conv_down_2
        DoubleConv(2*nfilters,4*nfilters), # double_conv_down_3
        DoubleConv(4*nfilters,8*nfilters), # double_conv_down_4
        DoubleConv(8*nfilters,16*nfilters), # double_conv_down_5
        DoubleConv(16*nfilters,8*nfilters), # double_conv_up_1
        DoubleConv(8*nfilters,4*nfilters), # double_conv_up_2
        DoubleConv(4*nfilters,2*nfilters), # double_conv_up_3
        DoubleConv(2*nfilters,nfilters), # double_conv_up_4
        UpSample(16*nfilters,8*nfilters), # upsample_1
        UpSample(8*nfilters,4*nfilters), # upsample_2
        UpSample(4*nfilters,2*nfilters), # upsample_3
        UpSample(2*nfilters,nfilters), # upsample_4
        PoolDrop(), # pooldrop
        Conv((1, 1), nfilters=>nlabels, sigmoid) # conv_1d
    )

end

function (u::Unet)(x::AbstractArray{T}) where T

    c1 = u.double_conv_down_1(x)
    p1 = u.pooldrop(c1)

    c2 = u.double_conv_down_2(p1)
    p2 = u.pooldrop(c2)

    c3 = u.double_conv_down_3(p2)
    p3 = u.pooldrop(c3)

    c4 = u.double_conv_down_4(p3)
    p4 = u.pooldrop(c4)

    c5 = u.double_conv_down_5(p4)

    u6 = u.upsample_1(c5,c4)
    c6 = u.double_conv_up_1(u6)

    u7 = u.upsample_2(c6,c3)
    c7 = u.double_conv_up_2(u7)

    u8 = u.upsample_3(c7,c2)
    c8 = u.double_conv_up_3(u8)

    u9 = u.upsample_4(c8,c1)
    c9 = u.double_conv_up_4(u9)

    output = u.activation(c9)

    return output

end