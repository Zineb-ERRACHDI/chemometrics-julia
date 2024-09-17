# Import necessary packages
using JLD2, CairoMakie, StatsBase,GLMakie
using Jchemo,JchemoData
using Flux
using Flux:params,train!,DataLoader
using Distributions,Random
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

# Assemble the training data using a DataLoader
batch_size=256
data=DataLoader((Xtrainst',ytrain), batchsize = batch_size,shuffle=false)

##DNN model 
# Define the DNN model with specified weight initialization
K_INIT =Flux.kaiming_normal(MersenneTwister(0);gain=sqrt(2))#the initialisation function
model = Chain(
    Dense(100=>36, elu;init=K_INIT,bias=true ),
    Dropout(0.2),
    Dense(36 => 18, elu;init=K_INIT,bias=true),
    Dropout(0.2),
    Dense(18 => 12, elu;init=K_INIT,bias=true),
    Dropout(0.2),
    Dense(12 => 1)
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

# Initialize the optimizer
lr=0.01*(batch_size/256)
opt = ADAM(lr)
param = params(model) 

# Train the model for 1500 epochs
epochs = 1:1500
loss_train=[]
loss_test=[]
for i in epochs
	train!(lossR,param,data, opt)
    push!(loss_train, lossR(Xtrainst',ytrain))
    push!(loss_test, lossR(xtestst',ytest))
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

# Plot predictions
plotxy(vec(pred), ytest; color = (:red, .5), step = 2,
    bisect = true, xlabel = "Predicted", 
    ylabel = "Observed (Test)",title="Prediction based on DNN model").f

# Loss graph
plot([loss_train[1:200], loss_test[1:200]], label=["Train Loss" "Test Loss"], xlabel="epoch", ylabel="Loss", title="Model loss", lw=2)

#dnn Coefficients
dnn= vec(model(xtestst'))
x_subset_dnn=xtestst[1:100, :]#substract the first 100 spectra
y_subset_dnn=dnn[1:100, :] #substract the first 100 predicted values
num_rows = size(x_subset_dnn, 1)
num_columns = size(x_subset_dnn, 2)

# Initialize a 2D array (matrix) filled with zeros
w_dnn = zeros(num_rows, num_columns)
# Define epsilon
epsilon = 1E-4
for i in 1:100
    # Create a copy of the input data with epsilon perturbation on feature i only
    x_pert_dnn = copy(x_subset_dnn)
    x_pert_dnn[:, i] .+= epsilon  # Apply epsilon perturbation to feature i

    # Compute new model predictions where the input is the locally perturbed spectra
    y_pred_pert_dnn = vec(model(x_pert_dnn'))

    # Compute the regression coefficients and store them in w_cnn
    w_dnn[:, i] .= (y_pred_pert_dnn[:, 1]- y_subset_dnn[:, 1]) / epsilon
end

# Create a plot of regression coefficients
f,ax=plotsp(w_dnn, wl_num;
    xlabel="Index of wavelength", ylabel=L"w_i",
    title="DNN regression coefficients",resolution = (1000, 350))
hlines!(ax,0,color=:black)
f