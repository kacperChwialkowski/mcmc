import random
import math
import numpy as np
from autograd import grad
__author__ = 'kcx'

def gibbs(N=50000,thin=10):
    x=0
    y=0
    xvex = np.zeros(N)
    yvec  = np.zeros(N)

    for i in range(N):
        for j in range(thin):
            x=random.gammavariate(3,1.0/(y*y+4))
            y=random.gauss(1.0/(x+1),1.0/math.sqrt(2*x+2))
        xvex[i] = x
        yvec[i] = y
    return xvex,yvec



def almost_density(x,y,k=10):
    return k*x**2*np.exp(-x*y**2 - y**2 + 2*y - 4*y)


grad_almost = grad(almost_density)

def g