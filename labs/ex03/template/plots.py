# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
#from build_polynomial import build_poly

#NOTE we copied this from our python notebook
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    basis_matrix = np.zeros(shape=(len(x), degree+1))
    for i in range(0,len(x)):
        for j in range(0,degree+1):
            basis_matrix[i,j] = x[i]**j 
    return basis_matrix

def plot_fitted_curve(y, x, weights, degree, ax):
    """plot the fitted curve."""
    ax.scatter(x, y, color='b', s=12, facecolors='none', edgecolors='r')
    xvals = np.arange(min(x) - 0.1, max(x) + 0.1, 0.1)
    tx = build_poly(xvals, degree)
    f = tx.dot(weights)
    ax.plot(xvals, f)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial degree " + str(degree))
