# InverseProblem

# Intended to provide more stable regularization than Lasso or Ridge

ip.optimalk(A) #this will print out optimal k

ip.invert(A,be,k,l) #this will invert your A matrix, where be is noisy be, k is the no. of iterations, and lambda is your dampening effect (best set to 1)
