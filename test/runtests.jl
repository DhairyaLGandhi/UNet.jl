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

  test_img = zeros(UInt8,(256, 256))
  Images.save("test.png", test_img )

  println(size(test_img))
  println(typeof(test_img))
  
  input = Float32.(reshape(channelview(load("./test/trainingdata/input.png")), 256, 256, 3, 1))

  
  target = reshape(load("./test/trainingdata/target.png"), 256, 256, 1, 1)
  ulabels = sort(unique(target))
  itarget = Float32.(map(v -> findall(ulabels .== v)[1],target))

  data = Iterators.repeated((input, itarget), 500)

  println(typeof(input))
  println(typeof(itarget))
  
  u = Unet(3, 7)
  opt = ADAM()
  function loss(x,y)
    l = Flux.crossentropy(u(x),y)
    println(l)
    return l
  end
  
  Flux.train!(loss, Flux.params(u), data, opt)


  probs = dropdims(u(input); dims = 4)
  maxprob, cartindx = findmax(probs; dims = 3)
  pred = dropdims(map(v -> v[3], cartindx); dims = 3)
  pred = UInt8.(30*pred)
  
  println(size(pred))
  println(typeof(pred))



  Images.save("test.png", pred)

  @save "unet3.jld" u

  maximum

end