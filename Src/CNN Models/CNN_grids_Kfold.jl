# Import necessary packages
using JLD2, CairoMakie, StatsBase,GLMakie
using Jchemo,JchemoData
using Flux:params,train!,DataLoader,throttle
using Statistics,Random
using Flux
using BSON
using Plots:plot
using MLDataUtils
using CUDA
# Ensure reproducibility by setting a random seed
Random.seed!(0)
## Import necessary functions
include("C:/Zinebjl/function/grid_Kfold_cnn.jl")

# Load data from the specified JLD2 file
db = "C:/Zinebjl/Data/wheatkernels.jld2"
@load db dat;
Xtrain = dat.Xtrain
Ytrain = dat.Ytrain
ytrain=Ytrain.prot
Xtest = dat.Xtest
Ytest = dat.Ytest
ytest=Ytest.prot

# Extract variable names (wl) and convert to numbers (wl_num)
wl = names(Xtrain)
wl_num = parse.(Float64, wl)

# Preprocess the data detrend followed by SNV
Xtrainp=snv(detrend(Xtrain; pol = 1))
Xtestp=snv(detrend(Xtest; pol = 1))

# Standardize the data
xmeans=colmean(Xtrainp)
xscales=colstd(Xtrainp)
Xtrainst=cscale(Xtrainp,xmeans,xscales)
colstd(Xtrainst)# Make sure the data is Standardized
xtestst=cscale(Xtestp,xmeans,xscales)

best_hyperparameters,best_score=grid_kfold(Xtrainst,ytrain,4)

# Define the perfect model using the best hyperparameters
K_INIT =Flux.kaiming_normal(MersenneTwister(0);gain=sqrt(2))#the initialisation function
model = Chain(
    x->reshape(x,size(x, 1),1, size(x, 2)),
    Conv((best_hyperparameters[6],),1 => best_hyperparameters[5],elu;init=K_INIT,bias=true,pad = SamePad(),stride = 1),
     # Adjusting the input size for the first Dense layer
    Flux.flatten,
    Dense(best_hyperparameters[5]*100 => best_hyperparameters[1][1], elu;init=K_INIT,bias=true),
    Dropout(best_hyperparameters[4]),
    Dense(best_hyperparameters[1][1] => best_hyperparameters[1][2], elu;init=K_INIT,bias=true),
    Dropout(best_hyperparameters[4]),
    Dense(best_hyperparameters[1][2] => best_hyperparameters[1][3], elu;init=K_INIT,bias=true),
    Dropout(best_hyperparameters[4]),
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
