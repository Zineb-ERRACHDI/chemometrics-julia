function grid_search_cnn(X,Y)
    # Selection of validation data
    ntrain = nro(X)
    pct = .20
    nval = Int64.(round(pct * ntrain))
    s = sample(1:ntrain, nval; replace = false)
    Xcal = rmrow(X, s) 
    ycal = rmrow(Y, s) 
    Xval = X[s, :] 
    yval = Y[s]
    # Define hyperparameters for grid search
    hidden_layer_sizes = [(36, 18, 12), (48, 24, 12), (24, 12, 6)]
    batch_sizes = [50, 100, 200,256]
    # Initialize variables to store the best hyperparameters and validation score
    best_hyperparameters = (36, 18, 12)
    best_score = Inf
    lr=0
    model=Chain()
    for hidden_layers in hidden_layer_sizes
        for batch_size in batch_sizes
            # Define the model architecture
            K_INIT =Flux.kaiming_normal(MersenneTwister(0);gain=sqrt(2))
            model = Chain(
            x->reshape(x,size(x, 1),1, size(x, 2)),
            Conv((5,),1 => 1,elu;init=K_INIT,bias=true,pad = SamePad(),stride = 1),
            # Adjusting the input size for the first Dense layer
            Flux.flatten,
            Dense(100 => hidden_layers[1], elu;init=K_INIT,bias=true),
            Dropout(0.3),
            Dense(hidden_layers[1] => hidden_layers[2], elu;init=K_INIT,bias=true),
            Dropout(0.1),
            Dense(hidden_layers[2] => hidden_layers[3],  elu;init=K_INIT,bias=true),
            Dropout(0.1),
            Dense(hidden_layers[3] => 1)
            )
            param=params(model)

            # Assemble the training data
            data = DataLoader((Xcal', ycal), batchsize = batch_size, shuffle = false)
                    
            #the loss function with l2_regularization
            loss(x, y)= Flux.mse(vec(model(x)), y)+sum(w -> sum(w .^ 2), Flux.params(model)) *(0.003/2)
                     
            # Initialize the optimizer
            lr=0.01*(batch_size/256)
            opt = Adam(lr)

            # Train the model
            epochs = 1:1500
            for i in epochs
                train!(loss,param,data, opt)
            end

            # Evaluate the model on the validation set
            pred = vec(model(Xval'))
            validation_score = rmsep(pred,yval)[1]

            # Determine if the combination of hyperparameters is better than the previous one
            if validation_score < best_score
                best_score = validation_score
                best_hyperparameters = (hidden_layers, lr, batch_size)
            end
        end
    end
    return best_hyperparameters,best_score
end