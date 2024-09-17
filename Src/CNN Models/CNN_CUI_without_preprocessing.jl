# Import necessary packages
using JLD2, CairoMakie, StatsBase,GLMakie
using Jchemo,JchemoData
using Flux:params,train!,DataLoader,throttle
using Statistics
using Flux
using Random
using BSON
using Plots:plot

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

# Assemble the training data
batch_size=256
data =DataLoader((Xtrainst',ytrain), batchsize =batch_size,shuffle=false)

##CNN model 
# Define the CNN model with specified weight initialization
K_INIT =Flux.kaiming_normal(MersenneTwister(0);gain=sqrt(2))#the initialisation function
model = Chain(
    x->reshape(x,size(x, 1),1, size(x, 2)),
    Conv((5,),1 => 1,elu;init=K_INIT,bias=true,pad = SamePad(),stride = 1), 
     # Adjusting the input size for the first Dense layer
    Flux.flatten,
    Dense(100=>36, elu;init=K_INIT,bias=true),
    Dropout(0.3),
    Dense(36=>18, elu;init=K_INIT,bias=true),
    Dropout(0.1),
    Dense(18=>12, elu;init=K_INIT,bias=true),
    Dropout(0.1),
    Dense(12=>1)
)

# Define the loss function with L2 regularization
function lossR(x, y)
    y_pred =vec(model(x))
    mse_loss = Flux.mse(y_pred, y)
    λ = 0.003
    B = λ / 2
    l2_regularization = sum(w -> sum(w .^ 2), Flux.params(model)) * B
    return mse_loss+ l2_regularization
end

# Define the optimizer
lr=0.01*(batch_size/256)
opt = ADAM(lr)
param=Flux.params(model)

# Train the model for 1500 epochs
epochs = 1:1500
loss_train=[]
loss_test=[]
for epoch in epochs
    @info "Epoch $epoch"
    Flux.train!(lossR, param, data, opt ,cb = throttle(() -> println(lossR(Xtrainst',ytrain)), 10))
    push!(loss_train, lossR(Xtrainst',ytrain))
    push!(loss_test, lossR(xtestst',ytest))
end

# Predictions on the train set
pred1 = vec(model(Xtrainst'))
rmsep(pred1, ytrain)
sep(pred1, ytrain)
bias(pred1, ytrain)

# Predictions on the test set
pred = vec(model(xtestst'))
rmsep(pred, ytest)
sep(pred, ytest)
bias(pred, ytest)

#loss graph
plot([loss_train, loss_test], label=["Train Loss" "Test Loss"], xlabel="epoch", ylabel="Loss", title="Model loss", lw=2)

#cnn Coefficients
cnn= vec(model(xtestst'))
x_subset_cnn=xtestst[1:100, :]#substract the first 100 spectra
y_subset_cnn=cnn[1:100, :]#substract the first 100 predicted values
num_rows = size(x_subset_cnn, 1)
num_columns = size(x_subset_cnn, 2)

# Initialize a 2D array (matrix) filled with zeros
w_cnn = zeros(num_rows, num_columns)
# Define epsilon
epsilon = 1E-4
for i in 1:100
    # Create a copy of the input data with epsilon perturbation on feature i only
    x_pert_cnn = copy(x_subset_cnn)
    x_pert_cnn[:, i] .+= epsilon  # Apply epsilon perturbation to feature i
    # Compute new model predictions where the input is the locally perturbed spectra
    y_pred_pert_cnn = vec(model(x_pert_cnn'))

    # Compute the regression coefficients and store them in w_cnn
    w_cnn[:, i] .= (y_pred_pert_cnn[:, 1]- y_subset_cnn[:, 1]) / epsilon
end


# Create a plot of regression coefficients 
f,ax=plotsp(w_cnn, wl_num; 
    xlabel="Index of wavelength", ylabel=L"w_i",
    title="CNN regression coefficients",resolution = (1000, 350))
hlines!(ax,[0],color=:black)
f