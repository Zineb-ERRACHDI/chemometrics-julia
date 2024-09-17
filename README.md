# chemometrics-julia

# NIR Calibration Using CNNs, PLSR, and DNN

This project explores the use of various machine learning techniques for Near Infrared (NIR) calibration, focusing on predicting protein content in soft wheat samples. The models implemented include Partial Least Squares Regression (PLSR), Deep Neural Networks (DNN), and Convolutional Neural Networks (CNN).

## Project Overview
The goal of this project is to develop, compare, and evaluate multivariate regression models for NIR calibration using Julia. The models predict the protein content of wheat samples based on spectral data.

## Models Implemented
- **PLSR**: A well-established technique in chemometrics for spectroscopic data analysis.
- **DNN**: A deep neural network with several fully connected layers.
- **CNN**: A convolutional neural network designed to automate the preprocessing of spectroscopic data.

## Installation

Clone the repository and install the required Julia packages.

```bash
git clone https://github.com/Zitech-iav/chemometrics-julia.git
cd chemometrics-julia

# Install dependencies via Julia's package manager
# Using Flux.jl for neural networks
using Pkg
Pkg.add("Flux")
