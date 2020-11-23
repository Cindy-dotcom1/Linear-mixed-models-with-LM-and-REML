import pandas
import numpy as np

import statsmodels.formula.api as smf

from scipy.optimize import minimize

# Create dataframe :

x = np.array([1, 2, 3, 4, 5, 6, 7])
Y = np.array([3])
a = np.array(["n", "i", "c"])
df = pandas.DataFrame({'ingot': np.repeat(x, np.repeat(Y, [7])), 'metal': ["n", "i", "c", "n", "i", "c", "n", "i", "c", "n", "i", "c", "n", "i", "c", "n", "i", "c", "n", "i", "c"], 'pres': [
                      67, 71.9, 72.2, 67.5, 68.8, 66.4, 76, 82.6, 74.5, 72.7, 78.1, 67.3, 73.1, 74.2, 73.2, 65.8, 70.8, 68.7, 75.6, 84.9, 69]})
df

# Linear mixed model with maximum likelihood estimation :
md = smf.mixedlm('pres ~ metal', df, groups=df.ingot, re_formula="~df.metal")
mdf = md.fit(reml=False) # We use reml = False to use maximum likelooh estimation rather than reml
print(mdf.summary()) # Regression results (particularly the coefficients of the regression)

# Manual estimation of the coefficients :
    
def Beta_estimated(x):
    Y = df.pres # ingot
    X = np.array([(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 1, 0, 0, 1, 0, 0, 1, 0,
                                                                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0), (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)]) # Design matrix X
    Z = np.mat('[1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;0.,0.,0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.; 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0.,0.,0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.]') # Design matrix Z
    G = x[0] * np.identity(7)
    R = x[1] * np.identity(21)
    V = (Z.T@G @ Z)+R
    Beta = np.linalg.inv(X@np.linalg.inv(V)@X.T)@X@np.linalg.inv(V)@Y #Manual estimation of the coefficients
    # BEWARE: never do this inv step !!! use a linear solver instead of a matrix inversion...
    return(Beta) 


Beta_estimated(np.array([1, 1]))

# Optimisation :
    
def optim(x):
    Y = df.pres # ingot
    X = np.array([(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 1, 0, 0, 1, 0, 0, 1, 0,
                                                                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0), (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)]) # Design matrix X
    Z = np.mat('[1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;0.,0.,0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.; 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0.,0.,0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.]') # Design matrix Z
    G = x[0] * np.identity(7)
    R = x[1] * np.identity(21)
    V = (Z.T@G @ Z)+R
    Beta = np.linalg.inv(X@np.linalg.inv(V)@X.T)@X@np.linalg.inv(V)@Y # Manuel estimation of the coefficients
    y = np.atleast_2d(Y)
    loglike = np.log(np.linalg.det(V)) + (y - (X.T @ Beta)
                                          )@(np.linalg.inv(V)) @ (y-X.T@Beta).T # Log likelihood function
    return(loglike[0, 0])


minimize(optim, x0=np.array([1, 1]), tol=0.0000000000001) # We minimise the log likelihood function for the estimation of the parameters variance_ingot and variance_residuals

# Linear mixed model by REML :
    
md = smf.mixedlm('pres ~ metal', df, groups=df.ingot, re_formula="~df.metal")
mdf = md.fit() # We use reml = True (default argument) to use REML
print(mdf.summary()) # Regression results (particularly the coefficients of the regression)

# Optimisation :
def optim_REML(x):
    Y = df.pres
    X = np.array([(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 1, 0, 0, 1, 0, 0, 1, 0,
                                                                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0), (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)]) # Design matrix X
    Z = np.mat('[1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;0.,0.,0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.; 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0., 0., 0., 0., 0., 0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.,0.,0.,0.;0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.,1.,1.]') # Design matrix Z
    G = x[0] * np.identity(7)
    R = x[1] * np.identity(21)
    V = (Z.T@G @ Z)+R
    Beta = np.linalg.inv(X@np.linalg.inv(V)@X.T)@X@np.linalg.inv(V)@Y # Coefficients of the regression
    y = np.atleast_2d(Y)
    loglike = np.log(np.linalg.det(V)) + np.log(np.linalg.det(X@np.linalg.inv(V)
                                                              @ X.T)) + (y - (X.T @ Beta))@(np.linalg.inv(V)) @ (y-X.T@Beta).T # Log likelihood function
    return loglike[0, 0]


minimize(optim_REML, x0=np.array([5, 6]), tol=0.0000000000001) # We minimise the log likelihood function for the estimation of the parameters variance_ingot and variance_residuals
