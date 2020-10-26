function expand_dims(x::AbstractArray{T},n::Int) where T
    return reshape(x,ones(Int64,n)...,size(x)...) 
end

function squeeze(x::AbstractArray{T}) where T
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end

#Base.Float32(x::ForwardDiff.Dual{Nothing,Float32,1}) = x