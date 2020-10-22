# function accuracy(data_loader, model)
#     acc = 0
#     for (x,y) in data_loader
#         acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
#     end
#     acc/length(data_loader)
# end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y; dims=3)
    end
    l/length(dataloader)
end



function train(train_dir_input, train_dir_target, test_dir_input, test_dir_target, nepochs, numfiles, batchsize, lr, lr_drop_rate, lr_step, model_dir)

    #initialize datasets
    train_dataset = initialize_dataset("train"; input_dir = train_dir_input, target_dir = train_dir_target)
    test_dataset = initialize_dataset("test"; input_dir = test_dir_input, target_dir = test_dir_target)
    
    # Construct model
    m = Unet(nchannels, nfeatures) |> gpu

    function loss(x,y)
        logitcrossentropy(m(x), y; dims=3)
    end

    # Load testing data 
    test_batch_input_files, test_batch_target_files = grab_random_files(test_dataset, 10; drop_processed = false)
    xtest, ytest, wtrain = load_files(test_batch_input_files, test_batch_target_files)

    @show typeof(xtest)
    @show size(xtest)
    @show typeof(ytest)
    @show size(ytest)

    test_data = DataLoader(xtest |> gpu, ytest |> gpu, batchsize=batchsize) |> gpu

    evalcb = () -> @show(loss_all(test_data, m))

    opt = ADAM(lr)

    for i = 1:nepochs
        if i % lr_step == 0
            opt.eta = maximum([1e-6, opt.eta*lr_drop_rate])
            @info "New learning rate $(opt.eta)"
        end
        while train_dataset.input_num_files > 0

            # Load training data 
            train_batch_input_files, train_batch_target_files = grab_random_files(test_dataset, numfiles)
            xtrain, ytrain, wtrain = load_files(train_batch_input_files, train_batch_target_files)

            @show typeof(xtrain)
            @show size(xtrain)
            @show typeof(ytrain)
            @show size(ytrain)
            train_data = DataLoader(xtrain |> gpu, ytrain |> gpu, batchsize=batchsize, shuffle=true) |> gpu

            @info "Training..."
            Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

        end

        #re-initialize dataset after all files processed
        train_dataset = initialize_dataset("train"; input_dir = train_dir_input, target_dir = train_dir_target)

        test_accuracy = loss_all(test_data, m)

        #print out accuracies
        @info "Accuracy on a testing set $(test_accuracy)"

        acc = string(test_accuracy)
        acc = acc[1:min(6,length(acc))]

        serialize(joinpath(model_dir,"model_epoch_$(i)_accuracy_$(acc).jls"), m |> cpu)
    end
  
end