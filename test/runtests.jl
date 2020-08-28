using Test, UNet
using UNet.Flux, UNet.Flux.Zygote
using Base.Iterators
using FileIO
using Images
using JLD


# @testset "Inference" begin

#   for ch in (1,3)
#     u = Unet(ch)
#     ip = rand(Float32, 256, 256, ch, 1)

#     @test size(u(ip)) == size(ip)
#   end
# end

# @testset "Variable Sizes" begin

#   u = Unet()
#   # test powers of 2 don't throw and return correct shape
#   for s in (64, 128, 256)
#     ip = rand(Float32, s, s, 1, 1)
#     @test size(u(ip)) == size(ip)
#   end

#   broken_ip = rand(Float32, 299, 299, 1, 1)
#   @test_throws DimensionMismatch size(u(broken_ip)) == size(broken_ip)
# end

# @testset "Gradient Tests" begin
#   u = Unet()
#   ip = rand(Float32, 256, 256, 1,1)
#   gs = gradient(Flux.params(u)) do
#     sum(u(ip))
#   end

#   @test gs isa Zygote.Grads
# end


@testset "Training test" begin

  input = Float32.(reshape(channelview(load("./test/trainingdata/input.png")), 256, 256, 3, 1))
  target = reshape(load("./test/trainingdata/target.png"), 256, 256, 1, 1)
  ulabels = sort(unique(target))
  itarget = Int32.(map(v -> findall(ulabels .== v)[1],target))

  onehottarget = zeros(Int32, 256, 256, 7, 1)

  for i=1:256
    for j=1:256
      k = itarget[i, j, 1, 1]
      onehottarget[i, j, k, 1] = 1
    end
  end

  println(itarget[125, 125, 1, 1])
  println(onehottarget[125, 125, :, 1])

  data = Iterators.repeated((input, itarget), 1)

  u = Unet(3, 7)

  function loss(x,y)
      Flux.logitcrossentropy(u(x), y; dims=3)
  end

  nEpochs = 1000
  lr = 0.001
  @show opt = ADAM(lr)
  for i =1:nEpochs
    Flux.train!(loss2, Flux.params(u), [(input, onehottarget)], opt)
    if i % 10 == 0
      opt.eta = maximum([1e-5, opt.eta/2.0])
      @info "New LR $(opt.eta)"

      probs = dropdims(u(input); dims = 4)
      maxprob, cartindx = findmax(probs; dims = 3)
      pred = dropdims(map(v -> v[3]-1, cartindx); dims = 3)
      
      # @show itarget[100:105, 100:105, 1, 1]
      # @show pred[100:105, 100:105]
    
      pred = UInt8.(40*pred)
      Images.save("test_$(i).png", pred)
    end
  end

  @save "unet.jld" u

end