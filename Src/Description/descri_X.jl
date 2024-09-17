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
# Print dimensions of the training set
println("Train set dimension X Y = ", size(Xtrain), size(Ytrain))


# Visualize the variable names
wl = names(Xtrain)
wl_num = parse.(Float64, wl)


# Compute statistics on the data Xtrain and Xtest
summ(Xtrain)
summ(Xtest)

# Visualize spectra using CairoMakie
plotsp(Xtrain, wl_num; nsamp = 20, 
    xlabel = "λ(nm)", ylabel="X",
    title="20 spectra from the train sample").f

# Perform Principal Component Analysis (PCA)
fm = pcasvd(Xtrain, nlv = 10) ; 
pnames(fm)
T = fm.T
# Display summary statistics of the PCA results
res = summary(fm, Xtrain) ;
pnames(res)
z = res.explvarx
# Plot the explained variance by Principal Components
plotgrid(z.lv, 100 * z.pvar; step = 1,
    xlabel = "Nombre de scores", ylabel = "% variance expliquées").f

# Plot the scores of the first and second Principal Components
i = 1
plotxy(T[:, i:(i +1)]; color = :red,
    xlabel = string("PC",i), ylabel = string("PC",1+i)).f
