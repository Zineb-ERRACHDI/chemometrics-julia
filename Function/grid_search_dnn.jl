function grid_search_dnn(X, Y)
    # Selection of validation data
    ntrain = nro(X)
    pct = .20
    nval = Int64.(round(pct * ntrain))
    s = sample(1:ntrain, nval; replace = false)
    Xcal = rmrow(X, s) 
    ycal = rmrow(Y, s) 
    Xval = X[s, :] 
    yval = Y[s]
    # Définir les hyperparamètres pour la recherche par grille
    hidden_layer_sizes = [(36, 18, 12), (48, 24, 12), (24, 12, 6)]
    batch_sizes = [50, 100, 200,256]
    learning_rate = [0.1, 0.01, 0.001]
    dropout_rate=[0.1, 0.2, 0.3]
    # Initialiser les variables pour stocker les meilleurs hyperparamètres et le score de validation
    best_hyperparameters = (36, 18, 12)
    best_score = Inf
    lr = 0
    model = Chain()
    
    for hidden_layers in hidden_layer_sizes
        for batch_size in batch_sizes
            for lr in learning_rate
                for dr in dropout_rate
                    # Définir l'architecture du modèle
                    K_INIT = Flux.kaiming_normal(MersenneTwister(0); gain = sqrt(2))
                    model = Chain(
                            Dense(100 => hidden_layers[1], elu; init = K_INIT, bias = true),
                            Dropout(dr),
                            Dense(hidden_layers[1] => hidden_layers[2], elu; init = K_INIT, bias = true),
                            Dropout(dr),
                            Dense(hidden_layers[2] => hidden_layers[3], elu; init = K_INIT, bias = true),
                            Dropout(dr),
                            Dense(hidden_layers[3] => 1)
                        )
                    param = params(model)
                        # Assembler les données d'entraînement
                    data = DataLoader((Xcal', ycal), batchsize = batch_size, shuffle = true)

                        # Fonction de perte avec régularisation L2
                    loss(x, y) = Flux.mse(vec(model(x)), y) + sum(w -> sum(w .^ 2), Flux.params(model)) * (0.003 / 2)

                        # Initialiser l'optimiseur
                    opt = Adam(lr)

                        # Entraîner le modèle
                    epochs = 1:1500
                    for i in epochs
                        train!(loss, param, data, opt)
                    end

                        # Évaluer le modèle sur l'ensemble de validation
                    pred = vec(model(Xval'))
                    validation_score = rmsep(pred, yval)[1]

                        # Déterminer si la combinaison d'hyperparamètres est meilleure que la précédente
                    if validation_score < best_score
                        best_score = validation_score
                        best_hyperparameters = (hidden_layers, batch_size,lr,dr)
                    end
                end
            end
        end
    end
    return best_hyperparameters, best_score
end
