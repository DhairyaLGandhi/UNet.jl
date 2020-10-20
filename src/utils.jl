function _random_normal(shape...)
  return Float32.(rand(Normal(0.f0,0.02f0),shape...))
end

# function extract_bboxes(mask)
#   nth = last(size(mask))
#   boxes = zeros(Integer, nth, 4)
#   for i =1:nth
#     m = mask[:,:,i]
#     cluster = findall(!iszero, m)
#     if length(cluster) > 0	
#       Is = map(x -> [x.I[1], x.I[2]], cluster) |> x -> hcat(x...)'
#       x1, x2 = extrema(Is[:,1])
#       y1, y2 = extrema(Is[:,2])
#     else
#       x1 ,x2, y1, y2 = 0, 0, 0, 0
#     end
#       boxes[i,:] = [y1, x1, y2, x2]
#   end
#   boxes
# end

# function extract_bboxes(masks::AbstractArray{T,4}) where T
#   bs = []
#   for i in 1:size(masks, 4)
#     b = extract_bboxes(masks[:,:,:,i])
#     push!(bs, b)
#   end
#   reduce(vcat, bs)
# end

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
function squeeze(x) 
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end

# function bce(ŷ, y; ϵ=gpu(fill(eps(first(ŷ)), size(ŷ)...)))
#   l1 = -y.*log.(ŷ .+ ϵ)
#   l2 = (1 .- y).*log.(1 .- ŷ .+ ϵ)
#   l1 .- l2
# end

# function loss(x, y)
#   op = clamp.(u(x), 0.001f0, 1.f0)
#   mean(bce(op, y))
# end
