using StatsBase: sample, shuffle

"""
  stub -> <one of the dirs>/<img path> - _im/cr.tif
"""
function load_img(base::String, stub::String; rsize = (256,256))
  im = joinpath(base, stub * "_im.tif")
  cr = joinpath(base, stub * "_cr.tif")
  x, y = load(im), load(cr)
  x = imresize(x, rsize...)
  y = imresize(y, rsize...)
  x, y = channelview(x), channelview(y)
  x = reshape(x, rsize..., 1, 1)
  y = reshape(y, rsize..., 1, 1)
  x, y
end

"""
    `load_batch(base::String, name_template, target_template; n = 10, dir = nothing, rsize = (500,500))`

  `base`: Path to the base directory where the training dataset is stored

  templates: `name_template`(::String) filters the images in the path based on whether they contain the `name_template`
             `target_template`(::String) replaces the `name_template` with the `target_template` to look for corresponding masks

  `dir`: if `nothing`, all the directories will be sampled at random, else will pick up images from the specified directory. Currently supports taking only one directory as a `String`. 

  Returns a batch of `n` randomly sampled images
  and corresponding masks.
"""
function load_batch(base::String, name_template = "im",
                    target_template = "cr";
                    n = 10, channels = 1,
                    dir=nothing, rsize=(500,500))

  # TODO: templates should support regexes
  if dir isa Nothing
    dir = setdiff(readdir(base), ["zips"])
    dir = sample(dir)
  end

  imgs = filter(x -> occursin(name_template, x),
                readdir(joinpath(base, dir)))
  imgs = sample(imgs, n)
  masks = map(x -> replace(x, name_template=>target_template), imgs)
  
  # Ensure all the files exist, else try again, and error
  if !all(isfile.(masks))
    imgs = sample(imgs, n)
    masks = map(x -> replace(x, name_template=>target_template), imgs)
  end

  batch = zip(imgs, masks)

  x = zeros(Float32, rsize..., channels, n) # []
  y = zeros(Float32, rsize..., channels, n) # []
  for (i,(img, mask)) in enumerate(batch)
    img = load(joinpath(base, dir, img))
    mask = load(joinpath(base, dir, mask))
    img = imresize(img, rsize...)
    mask = imresize(mask, rsize...)
    img = channelview(img)
    mask = channelview(mask)
    img = reshape(img, rsize..., channels)
    mask = reshape(mask, rsize..., channels)
    
    x′ = @view x[:,:,:,i]
    x′ .= img

    y′ = @view x[:,:,:,i]
    y′ .= mask
  end

  x, y
end
