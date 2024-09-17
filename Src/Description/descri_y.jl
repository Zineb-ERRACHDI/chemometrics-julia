# Import necessary packages
using JLD2, CairoMakie, StatsBase,GLMakie
using Jchemo,JchemoData

# Load data from the specified JLD2 file
db ="C:/Zinebjl/Data/wheatkernels.jld2"
@load db dat;
Xtrain = dat.Xtrain
Ytrain = dat.Ytrain
ytrain=Ytrain.prot
Xtest = dat.Xtest
Ytest = dat.Ytest
ytest=Ytest.prot

# Plot the y-values for the training set
wl= 1:nro(ytrain)
plotxy(wl,ytrain;
    xlabel = "Sample number",ylabel="Proteines",
    title="Train y labels").f

# Plot the y-values for the test set
wlt= 1:nro(ytest)
plotxy(wlt,ytest;
    xlabel = "Sample number",ylabel="Proteines",
    title="Test y labels").f

# Compute statistics on the y training and test data  (Ytrain and Ytest)
summ(Ytrain)
summ(Ytest)
#the distribution of y train and test data
f = Figure(resolution = (500, 400))
ax = Axis(f[1, 1], xlabel = "Protein", ylabel = "Density",title="Distributions of Y train and test data")
GLMakie.density!(ax,ytrain; bins = 50, label = "Train", color = (:red, 0.5),bandwidth = 0.2)
GLMakie.density!(ax,ytest;  bins = 50, label = "Test", color = (:blue, 0.5),bandwidth = 0.2)
axislegend(position = :rt, framevisible = false)
f