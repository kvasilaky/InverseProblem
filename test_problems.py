'''
This script uses the InverseProblem package 
https://github.com/kathrynthegreat/InverseProblem
https://pypi.python.org/pypi?%3Aaction=pkg_edit&name=InverseProblem

InverseProblem.invert() is a function for the Generalized Tikhonov Problem uses a solution from an iterative ``plug and replug`` 
approach that passes by the noiseless solution. 

InverseProblem.invert() take three parameters A, the matrix, l, lambda the dampening factor, and k the iterations.  

'''

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import InverseProblem.functions as ip

#####################
#Example 1 Perturb b, a slight change in the observed data
#####################
A = np.matrix('.16, .10; .17 .11; 2.02 1.29')
#A is ill-conditioned, 
np.linalg.cond(A)
x=np.matrix('1;1')
b=A*x #noise free right hand side

#Now add noise to b of size ro
# dimension of A
r= np.matrix(A.shape)[0,0]
#Introduce a slight perturbation to observed data
nb= np.random.normal(0, .01, r) #compute vector of normal variants mean=0,std=0.1
#You'll see that the data output is bad, high variance, so no solution is really good
#nb= np.random.normal(0, 50, r) #compute vector of normal variants mean=0,std=50

#note that nb is 3 by 1, so we need to flip it
be=b+np.matrix(nb).transpose() #add noise
 
l=1
k=20

ip.test()
#Slight perturbation to b and the pseudo inverse sucks
np.linalg.pinv(A)*be
#Slight perturbation to b and the iterative approach rocks
ip.invert(A,be,k,l)

#####################
#Example 2 Perturb A
#####################
#This example is meant to show that the pseudo inverse is an unstable solution
#A is singular (deficient in rank). A_delta is almost singular. 
#The pseudo inverse solution is b=[1 0]. 
#When you perturb A slightly by one one millionth of a delta, the solution (x) change by a millionth fold
#But when you use the iterative appraoch, the solution does not change by much
# (Note: in this example we don't know the actual solution. We're just showing that Pseudo inverse sln is unstable)

A = np.matrix('1, 0; 0 0; 0 0')
A_delta = np.matrix('1, 0; 0 .00000001; 0 0')
#condition number as being (very roughly) the rate at which the solution, x, will change with respect to a change in b.
np.linalg.cond(A)
np.linalg.cond(A_delta)
b=np.matrix('1;1;1') #noise free right hand side


l=.01
k=100
ip.invert(A_delta,b,k,l)
np.linalg.pinv(A_delta)*b

#####################
#Example 3 Hilbert A
#####################

from scipy.linalg import hilbert
A=hilbert(3)
np.linalg.cond(A)
x=np.matrix('1;1;1')
b=A*x

#Now add noise to b of size ro
# dimension of A
r= np.matrix(A.shape)[0,0]
nb= np.random.normal(0, .50, r) #compute vector of normal variants mean=0,std=0.1
#nb= np.random.normal(0, 50, r) #compute vector of normal variants mean=0,std=50
#note that nb is 3 by 1, so we need to flip it
be=b+np.matrix(nb).transpose() #add noise

l=1
k=100


np.linalg.pinv(A)*be
ip.invert(A,be,k,l)

