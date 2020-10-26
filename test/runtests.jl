using Test, UNet
using UNet.Flux, UNet.Flux.Zygote
using Base.Iterators
using FileIO
using Images
using JLD

@testset "Overfitting test" begin

  input = Float32.(reshape(channelview(load("./test/testdata/input.png")), 256, 256, 3, 1))
  target = reshape(load("./test/testdata/target.png"), 256, 256, 1, 1)
  ulabels = sort(unique(target))
  itarget = Int32.(map(v -> findall(ulabels .== v)[1],target))

  onehottarget = zeros(Int32, 256, 256, 7, 1)

  for i=1:256
    for j=1:256
      k = itarget[i, j, 1, 1]
      onehottarget[i, j, k, 1] = k
    end
  end

  u = Unet(3, 7)

  function loss(x,y)
      Flux.logitcrossentropy(u(x), y; dims=3)
  end

  nEpochs = 1000
  lr = 0.01
  lr_step = 10
  lr_drop_factor = 0.95
  @show opt = ADAM(lr)

  for i =1:nEpochs
    Flux.train!(loss, Flux.params(u), [(input, onehottarget)], opt)
    if i % lr_step == 0
      opt.eta = maximum([1e-6, opt.eta*lr_drop_factor])
      @info "New LR $(opt.eta)"

      probs = dropdims(u(input); dims = 4)
      maxprob, cartindx = findmax(probs; dims = 3)
      pred = dropdims(map(v -> v[3]-1, cartindx); dims = 3)
      
      pred = UInt8.(30*pred)
      Images.save("test_$(i).png", pred)
    end
  end

  probs = dropdims(u(input); dims = 4)
  maxprob, cartindx = findmax(probs; dims = 3)
  pred = dropdims(map(v -> v[3]-1, cartindx); dims = 3)
  pred = UInt8.(30*pred)
  Images.save("./test/testdata/prediction.png", pred)

  @save "./test/testdata/unet_model.jld" u

end