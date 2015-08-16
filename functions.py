'''
define a function that takes a contaminated matrix, A, uses SVD to decompose it, and then gives the solution to Ax=b
Requires matrix A, vector b, parameters k&l, 

'''

def test():
    return (u'Actions demolish their alternatives. Experiments reveal them.')

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import this

def invert(A, b, k, l):

	U, s, V = np.linalg.svd(A, full_matrices=False) #compute SVD without 0 singular values

	#number of `columns` in the solution s, or length of diagnol
	c1=np.matrix(s).shape[1]
	S= np.diag(s)

	Si=S
	for i in range(0,c1):
		Si[i,i]=(1/Si[i,i]) - (1/Si[i,i])*(l/(l+Si[i,i]**2))**k
		#print Si

	#solution to x using noisy b and iterative T approach
	x1=V*Si*U.transpose()*b
	#return(u'This is the solution, X, to AX=b where A is ill-conditioned and b is noisy using a iterative Tikhonov regularization approach.')
	return(u'This is the solution, X, to AX=b where A is ill-conditioned and b is noisy using a iterative Tikhonov regularization approach.', x1) 

	
