function grid_kfold(X,Y,fold)
    ov = obsview((X', Y))
    folds = kfolds(ov, k = fold)
    model=Chain()
    best_score = Inf
    best_hyperparameters = (36, 18, 12)
    hidden_layer_sizes = [(36, 18, 12), (48, 24, 12), (24, 12, 6)]
    dropout_rate=[0.0,0.1,0.2,0.3,0.4]
    filtre=[1,2,3,4,5,6,7]
    filter_width=[3,4,5,6,7,8,9]
    for width in filter_width
        for fil in filtre
            for hidden_layers in hidden_layer_sizes
                for rate in dropout_rate
                    for i in 1:length(fold)
                        # Extract training and testing data
                        Xtrain_fold = folds[i][1][1]
                        ytrain_fold = folds[i][1][2]                 
                        Xtest_fold = folds[i][2][1]
                        ytest_fold = folds[i][2][2]
                        batch_size=size(Xtrain_fold,2)
                            # Corrected reshaping for DataLoader
                        data_fold = DataLoader((Xtrain_fold, ytrain_fold),batchsize=batch_size, shuffle=true)
                            # Initialize the model and optimizer for each fold
                        K_INIT =Flux.kaiming_normal(MersenneTwister(0);gain=sqrt(2))
                        model = Chain(
                                        x->reshape(x,size(x, 1),1, size(x, 2)),
                                        Conv((width,),1 => fil,elu;init=K_INIT,bias=true,pad = SamePad(),stride = 1),
                                        Flux.flatten,
                                        Dense(fil*100 => hidden_layers[1], elu; init = K_INIT, bias = true),
                                        Dropout(rate),
                                        Dense(hidden_layers[1] => hidden_layers[2], elu; init = K_INIT, bias = true),
                                        Dropout(rate),
                                        Dense(hidden_layers[2] => hidden_layers[3], elu; init = K_INIT, bias = true),
                                        Dropout(rate),
                                        Dense(hidden_layers[3] => 1)
                                    )|> gpu
                            #the loss function with l2_regularization
                        lossR(x, y)= Flux.mse(vec(model(x)), y)+sum(w -> sum(w .^ 2), Flux.params(model)) *(0.003/2)
                        param = Flux.params(model)
                        # Initialize the optimizer
                        lr=0.01*(batch_size/256)
                        opt = Adam(lr)

                            # Train the model for 1500 epochs
                        epochs = 1:1500
                        for epoch in epochs
                            Flux.train!(lossR, param, data_fold, opt)
                        end
                            # Evaluate the model on the validation set
                        pred = vec(model(Xtest_fold))
                        validation_score = rmsep(pred, ytest_fold)[1]

                            # Determine if the combination of hyperparameters is better than the previous one
                        if validation_score < best_score
                            best_score = validation_score
                            best_hyperparameters = (hidden_layers,lr, batch_size,rate,fil,width)
                        end
                    end
                end
            end
        end
    end
    return best_hyperparameters,best_score
end