
@with_kw mutable struct Dataset

    name::String = ""
    input_directory::String = "/input"
    input_prefix::String = "input_"
    input_files::Dict{Int,String} = Dict{Int,String}()
    input_files_keys::Array = []
    input_num_files::Int = 0

    target_directory::String = "/target"
    target_prefix::String = "target_"
    target_files::Dict{Int,String} = Dict{Int,String}()
    target_files_keys::Array = []
    target_num_files::Int = 0

    ini :: Function = initialize_dataset

end

function initialize_dataset(name; input_dir = input_dir::String, target_dir = target_dir::String)

    dataset = Dataset()

    dataset.name = name

    dataset.input_directory = input_dir
    input_files = readdir(dataset.input_directory)
    dataset.input_num_files = length(input_files)
    i = 1
    for f in input_files
        dataset.input_files[i] = joinpath(dataset.input_directory, f)
        i = i + 1
    end
    dataset.input_files_keys = sort(collect(keys(dataset.input_files)))

    dataset.target_directory = target_dir
    target_files = readdir(dataset.target_directory)
    dataset.target_num_files = length(target_files)
    i = 1
    for f in target_files
        dataset.target_files[i] = joinpath(dataset.target_directory, f)
        i = i + 1
    end
    dataset.target_files_keys = sort(collect(keys(dataset.target_files)))

    # check if directories in sync
    @assert dataset.input_num_files == dataset.target_num_files "Number of input files is different from number of target files! Input: $(dataset.input_num_files) Target:$(dataset.target_num_files)"

    input_files = string.(map(v -> replace(v,dataset.input_prefix=>""),input_files))
    target_files = string.(map(v -> replace(v,dataset.target_prefix=>""),target_files))

    @assert insync(input_files, target_files) "Input and target directories are not in sync. Make sure each input image has corresponding target image!"

    return dataset
end

function insync(input_files, target_files)

    isinsync = true
    for f in input_files
        if !in(f, target_files)
            isinsync = false
            break
        end
    end

    return isinsync
end

function grab_random_files(dataset::Dataset, num_files::Int; drop_processed = true)

    idx = sample(dataset.input_files_keys, min(dataset.input_num_files, num_files)) #this is not unique, some files are duplicated
    input_files = []
    target_files = []

    for i in idx
        push!(input_files, dataset.input_files[i])
        push!(target_files, dataset.target_files[i])
        
    end

    if drop_processed
        for i in unique(idx)
            pop!(dataset.input_files,i)
            pop!(dataset.target_files,i)
        end
    end

    dataset.input_num_files = length(dataset.input_files)
    dataset.input_files_keys = collect(keys(dataset.input_files))

    dataset.target_num_files = length(dataset.target_files)
    dataset.target_files_keys = collect(keys(dataset.target_files))

    return input_files, target_files

end

function load_files(input_files::Array, target_files::Array)

    nfiles = length(input_files)

    s = size(permutedims(channelview(load(input_files[1])), [2,3,1]))
    data = zeros(s[1], s[2], s[3], nfiles)

    i = 1
    for file in input_files
        img = permutedims(channelview(load(file)), [2,3,1])
        @assert s == size(img) "Input images are not of the same size. Please check!"
        data[:,:,:,i] = img
        i = i + 1
    end

    nfiles = length(target_files)
    s = size(channelview(load(target_files[1]))) 
    onehotlabels = zeros(Int8, s[1], s[2], nfeatures, nfiles)
    itargets = zeros(Int16, s[1], s[2], nfiles)
    weights = zeros(nfeatures, nfiles)
    i = 1
    for file in target_files
        target = channelview(load(file))
        itargets[:,:,i] = Int16.(get_integer_intensity.(target))
        @assert s == size(target) "Input images are not of the same size. Please check!"
        onehotlabels[:,:,:,i], weights[:,i] = target_to_onehot(target, nfeatures)
        i = i + 1
    end

    #return convert(Array{Float32}, data), convert(Array{Int8}, onehotlabels), convert(Array{Float32}, weights)
    return data, onehotlabels, itargets
end

function get_integer_intensity(value::Normed{UInt8,8})
    return value.i
end

function target_to_onehot(target, nfeatures)

    s = size(target)
    onehottarget = zeros(Int32, s[1], s[2], nfeatures)
    ulabels = sort(unique(target)) #nfeatures defined in defaults.jl
    itarget = 1 .+ Int8.(get_integer_intensity.(target) ./ 30)
    
    weights = zeros(nfeatures)

    for i=1:s[1]
        for j=1:s[1]
        k = itarget[i, j, 1]
        weights[k] += 1
        end
    end

    weights = weights/maximum(weights)
    weights = 1 .+ (1 .- weights) .* 100
    weights = Int8.(round.(weights))

    for i=1:s[1]
        for j=1:s[1]
        k = itarget[i, j, 1]
        onehottarget[i, j, k] = weights[k] # this should be 1 but we'll put weights here to balnce things out 
        end
    end

    return onehottarget, weights

end 

