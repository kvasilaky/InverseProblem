'''
define a function that takes a contaminated matrix, A, uses SVD to decompose it, and then gives the solution to Ax=b
Requires matrix A, vector b, parameters k&l, 

'''

from __future__ import division  #make / floating point division
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import array
import scipy
from scipy import linalg


#import this

def invert(A, b, k, l):

	u, s, v = linalg.svd(A, full_matrices=False) #compute SVD without 0 singular values

	#number of `columns` in the solution s, or length of diagnol
	
	S = np.diag(s)
	sr, sc = S.shape          #dimension of


	for i in range(0,sc-1):
		if S[i,i]>0.00001:
			S[i,i]=(1/S[i,i]) - (1/S[i,i])*(l/(l+S[i,i]**2))**k

	x1=np.dot(v.transpose(),S)    #why traspose? because svd returns v.transpose() but we need v
	x2=np.dot(x1,u.transpose())
	x3=np.dot(x2,b)

	return(u'This is the solution, X, to AX=b where A is ill-conditioned and b is noisy using a iterative Tikhonov regularization approach.', x3) 



#Elbow computes the elbow of a convex function
def elbow(N):
    import numpy as np
    import matplotlib.pyplot as plt
    import array
    import scipy
    from scipy import linalg
    import matplotlib.pyplot as plt
    allCoord=N
    (nPoints,c)=allCoord.shape
    lastIndex=nPoints-1
    firstPoint=allCoord[0,:]
    lastPoint=allCoord[lastIndex,:]
    lineVec=np.subtract(lastPoint, firstPoint)              #vector from first to last point
    lineVecN = [x / np.linalg.norm(lineVec)for x in lineVec]    #divide by norm of lineVec to normalize
    lineVecN=np.array(lineVecN)
    vecFromFirst= np.subtract(allCoord,firstPoint)                     # vector between all points and first point
    vecFromFirstN=np.tile(lineVecN,(nPoints,1))
    #vecFromFirstN=vecFromFirstN[0,:,:]
    scalarProduct=np.empty((nPoints,))
    for i in range(nPoints):
        scalarProduct[i] = np.dot(vecFromFirst[i],vecFromFirstN[i])
    vecFromFirstParallel=scalarProduct[:,None]*lineVecN
    #vecFromFirstParallel=vecFromFirstParallel[0,:,:]
    vecToLine= vecFromFirst- vecFromFirstParallel
    distToLine = np.apply_along_axis(np.linalg.norm, 1,vecToLine )
    maxDist=np.amax(distToLine)
    ind = np.argmax(distToLine)
    plt.plot(allCoord[:,0],allCoord[:,1])
    plt.hold(True)
    plt.plot([ind], [allCoord[ind,1]], 'g.', markersize=20.0)
    plt.show()
    return ind

#A=np.matrix('1 3 11 0  -11 -15;18 55 209 15 -198 -277; -23 -33 144 532 259 82;9 55 405 437 -100 -285;3 -4 -111 -180 39 219;-13 -9  202 346  401  253')

def optimalk(A):
	r, c = A.shape
	x0=np.zeros((c,1))
	x=np.ones((c,1))
	b =A*x
	nb = np.random.normal(0, .1, (r,1)) #add vector of normal variants mean=0,std=0.1
	be = b+nb #add noise
	u, s, v = linalg.svd(A, full_matrices=False)
	s = np.diag(s)
	l=1
	l1=.1
	k=200
	sr, sc = s.shape
	#S = csr_matrix(S) 
	D=np.zeros((sr,k))
	D1=np.zeros((sr,k))
	N=np.zeros((k,sc))
	N1=np.zeros((k,sc))
	for j in range(0,k):
	    S0=np.copy(s)  #make a copy of s 
	    S1=np.copy(s)   #make copy of s (not reference)
	    for i in range(0,sc):
			S0[i,i]=(l/(l+S0[i,i]**2))**(j+1)
			S1[i,i]=(l1/(l1+S1[i,i]**2))**(j+1)
	    x1=np.dot(S0,u.transpose())
	    x2=np.dot(x1,be)
	    x2=np.reshape(x2,sr)
	    D[:,j]=x2
	    y1=np.dot(S1,u.transpose())
	    y2=np.dot(y1,be)
	    y2=np.reshape(y2,sr)
	    #y2=y2.ravel()    #conver 2d to 1d
	    D1[:,j]=y2
	    N[j,1]=np.linalg.norm(D[:,j], ord=None, axis=None)
	    N1[j,1]=np.linalg.norm(D1[:,j], ord=None, axis=None)
	    N1[j,0]=j
	    N[j,0]=j
	k=elbow(N)
	return('this is the optimal number of k iterations', k)



Golub=np.matrix('1 3 11 0  -11 -15;18 55 209 15 -198 -277; -23 -33 144 532 259 82;9 55 405 437 -100 -285;3 -4 -111 -180 39 219;-13 -9  202 346  401  253')
