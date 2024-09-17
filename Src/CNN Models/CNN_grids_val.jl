#!Given the very small dataset, using grid search with a train and validation split
# is unlikely to give good results especially in ANN.

# Import necessary packages
using JLD2, CairoMakie, StatsBase,GLMakie
using Jchemo,JchemoData
using Flux:params,train!,DataLoader,throttle
using Statistics,Random
using Flux
using BSON
using Plots:plot

## Import necessary functions
include("C:/Zinebjl/function/grid_search_cnn.jl")

# Ensure reproducibility by setting a random seed
Random.seed!(0)

# Load data from the specified JLD2 file
db = "C:/Zinebjl/Data/wheatkernels.jld2"
@load db dat;
Xtrain = dat.Xtrain
Ytrain = dat.Ytrain
ytrain=Ytrain.prot
Xtest = dat.Xtest
Ytest = dat.Ytest
ytest=Ytest.prot

# Standardize the data
xmeans=colmean(Xtrain)
xscales=colstd(Xtrain)
Xtrainst=cscale(Xtrain,xmeans,xscales)
colstd(Xtrainst)# Make sure the data is Standardized
xtestst=cscale(Xtest,xmeans,xscales)

# for hyperparameter tuning
best_hyperparameters,best_score=grid_search_cnn(Xtrainst, ytrain)

# Define the perfect model using the best hyperparameters
K_INIT =Flux.kaiming_normal(MersenneTwister(0);gain=sqrt(2))#the initialisation function
model = Chain(
    x->reshape(x,size(x, 1),1, size(x, 2)),
    Conv((5,),1 => 1,elu;init=K_INIT,bias=true,pad = SamePad(),stride = 1),
     # Adjusting the input size for the first Dense layer
    Flux.flatten,
    Dense(100 => best_hyperparameters[1][1], elu;init=K_INIT,bias=true),
    Dropout(0.3),
    Dense(best_hyperparameters[1][1] => best_hyperparameters[1][2], elu;init=K_INIT,bias=true),
    Dropout(0.1),
    Dense(best_hyperparameters[1][2] => best_hyperparameters[1][3], elu;init=K_INIT,bias=true),
    Dropout(0.1),
    Dense(best_hyperparameters[1][3] => 1))

# Define the loss function with L2 regularization
loss(x, y)= Flux.mse(vec(model(x)), y)+sum(w -> sum(w .^ 2), Flux.params(model)) *(0.003/2)

# Assemble the training data
data=DataLoader((Xtrainst',ytrain), batchsize = best_hyperparameters[3],shuffle=false);

# Initialize the optimizer
opt = Adam(best_hyperparameters[2])
param = params(model) 

# Train the model for 1500 epochs
epochs = 1:1500
losses = []
for i in epochs
	train!(loss,param,data, opt)
    push!(losses,loss(Xtrainst',ytrain));
end

# Predictions on the train set
pred1 = vec(model(Xtrainst'))
rmsep(pred1, ytrain)
sep(pred1, ytrain)
bias(pred1, ytrain)

# Predictions on the train set
pred = vec(model(xtestst'))
rmsep(pred, ytest)
sep(pred, ytest)
bias(pred, ytest)
