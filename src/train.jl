function dicecoefficient(data, model)
    #@info "Computing dice coefficient..."
    dice = 0
    x, y = data
    s = size(x)
    for i = 1:s[4]
        pred = probs_to_image(model, x[:,:,:,i:i])
        Images.save("test_$(i).png", UInt8.(pred |> cpu))
        dice += dice_coeff_loss(pred, y[:,:,i])
    end
    1 - dice/s[4]
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y; dims=3)
    end
    l/length(dataloader)
end

function probs_to_image(model, input)

    probs = dropdims(model(input); dims = 4)
    maxprob, cartindx = findmax(probs; dims = 3)
    pred = dropdims(map(v -> v[3]-1, cartindx); dims = 3)
      
    pred = Int16.(30*pred)

    return pred
end

function train(train_dir_input, train_dir_target, test_dir_input, test_dir_target, nepochs, numfiles, batchsize, lr, lr_drop_rate, lr_step, model_dir)

    #initialize datasets
    train_dataset = initialize_dataset("train"; input_dir = train_dir_input, target_dir = train_dir_target)
    test_dataset = initialize_dataset("test"; input_dir = test_dir_input, target_dir = test_dir_target)
    
    # Construct model
    #CUDA.allowscalar(false)
    m = Unet(nchannels, nfeatures) |> gpu
 
    function loss(x::AbstractArray{T}, y) where T
        return logitcrossentropy(m(x), y)
    end

    # Load testing data 
    test_batch_input_files, test_batch_target_files = grab_random_files(test_dataset, 1; drop_processed = false)
    xtest, ytest, wtest = load_files(test_batch_input_files, test_batch_target_files)

    test_data = DataLoader(xtest |> gpu, ytest |> gpu, batchsize=1, shuffle=true) |> gpu
    dice_test_data = (xtest, wtest) #|> gpu

    evalcb = () -> @info "Minibatch loss: $(loss_all(test_data, m))"

    #opt = ADAM(lr)
    opt = RMSProp(lr, 0.95)
    max_dice_coefficient = 0.0
    for i = 1:nepochs
        if i % lr_step == 0
            opt.eta = maximum([1e-4, opt.eta*lr_drop_rate])
            @info "New learning rate $(opt.eta)"
        end
        while train_dataset.input_num_files > 0

            # Load training data 

            train_batch_input_files, train_batch_target_files = grab_random_files(train_dataset, numfiles)
            xtrain, ytrain, wtrain = load_files(train_batch_input_files, train_batch_target_files)

            if length(train_batch_input_files) > batchsize
                train_data = DataLoader(xtrain |> gpu, ytrain |> gpu, batchsize=batchsize, shuffle=true) |> gpu
                #train_data = [(xtrain, ytrain)] |> gpu
                Flux.train!(loss, Flux.params(m), train_data, opt, cb = evalcb)
            else
                continue
            end
            #Flux.train!(loss, Flux.params(m), train_data, opt)
        end

        #re-initialize dataset after all files processed
        train_dataset = initialize_dataset("train"; input_dir = train_dir_input, target_dir = train_dir_target)

        test_accuracy = dicecoefficient(dice_test_data, m |> cpu)
        #test_accuracy = 0
        @info "Epoch $i, Dice coefficient on a testing set $(test_accuracy)"

        if test_accuracy > max_dice_coefficient
            @info "New best Dice coefficient!"
            max_dice_coefficient = test_accuracy
            acc = string(test_accuracy)
            acc = acc[1:min(6,length(acc))]

            serialize(joinpath(model_dir,"model_epoch_$(i)_accuracy_$(acc).jls"), m |> cpu)
        end
    end
  
end