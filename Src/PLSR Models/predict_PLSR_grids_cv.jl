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

# Extract variable names (wl) and convert to numbers (wl_num)
wl = names(Xtrain)
wl_num = parse.(Float64, wl)

# Preprocess data by detrend followed by SNV
Xtrainp=snv(detrend(Xtrain; pol = 1))
Xtestp= snv(detrend(Xtest; pol = 1))
# Visualize preprocessed spectra
plotsp(Xtrainp, wl_num; nsamp = 50,
    xlabel = "Wavelength (nm)", ylabel = "X",
    title="Preprocessed train set spectra: Detrend + SNV").f
plotsp(Xtestp, wl_num; nsamp = 50,
    xlabel = "Wavelength (nm)", ylabel = "X",
    title="Preprocessed test set spectra: Detrend + SNV").f

# PLS (Partial Least Squares)
K = 10 #number of folds
ntrain=nro(Xtrainp)
segm = segmkf(ntrain, K; rep = 50)

# Tuning the number of Latent Variables (LVs)
nlv = 0:20
rescv = gridcvlv(Xtrainp, ytrain; segm = segm, 
    score = rmsep, fun = plskern, nlv = nlv, 
    verbose = true) ;
pnames(rescv)
# Results for each replication
res_rep = rescv.res_rep

# Average results over the replications
res = rescv.res
plotgrid(res.nlv, res.y1; step = 2,
    xlabel   = "Nb. LVs", ylabel = "RMSEP").f

u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

# Variability of the performance between folds and replications
group = string.(res_rep.segm, "-", res_rep.repl)
plotgrid(res_rep.nlv, res_rep.y1, group; step = 2,
    xlabel = "Nb. LVs", ylabel = "RMSEP", leg = false).f

# Prediction using the optimal model
fm = plskern(Xtrainp, ytrain; nlv = res.nlv[u]) ;
# Predictions on the train set
pred1= Jchemo.predict(fm, Xtrainp).pred
rmsep(pred1, ytrain)
sep(pred1, ytrain)
bias(pred1, ytrain)


# Predictions on the test set
pred = Jchemo.predict(fm, Xtestp).pred
rmsep(pred, ytest)
sep(pred, ytest)
bias(pred, ytest)

# Plot predictions
plotxy(vec(pred), ytest; color = (:red, .5), step = 2,
    bisect = true, xlabel = "Predicted", 
    ylabel = "Observed (Test)", title="Prediction from PLS").f

# Residual analysis
residu = residreg(pred, ytest)
zr = vec(residu)
f,ax=plotxy(ytest, zr; resolution = (500, 400),
    color = (:red, .5),
    xlabel = "Observed (Test)", ylabel = "Residual")
hlines!(ax,0)
f

# PLS Coefficients
pls= Jchemo.predict(fm, Xtestp).pred
x_subset_pls=Xtestp[1:100, :]#substract the first 100 spectra
y_subset_pls=pls[1:100, :]#substract the first 100 predicted values
num_rows = size(x_subset_pls,1)
num_columns = size(x_subset_pls, 2)

# Initialize a 2D array (matrix) filled with zeros
w_pls = zeros(num_rows, num_columns)
# Define epsilon
epsilon = 1E-4
for i in 1:100
    # Create a copy of the input data with epsilon perturbation on feature i only
    x_pert_pls = copy(x_subset_pls)
    x_pert_pls[:, i] .+= epsilon  # Apply epsilon perturbation to feature i

    # Compute new model predictions where the input is the locally perturbed spectra
    y_pred_pert_pls = Jchemo.predict(fm, x_pert_pls).pred

    # Compute the regression coefficients and store them in w_pls
    w_pls[:, i] .= (y_pred_pert_pls[:, 1]- y_subset_pls[:, 1]) / epsilon
end

# Create a plot of regression coefficients
f,ax=plotsp(w_pls, wl_num; xlabel="Index of wavelength", 
     ylabel=L"w_i",title="PLS regression coefficients",
     resolution = (1000, 350), color=:brown)
hlines!(ax,0, color=:black)
f