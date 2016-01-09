import warnings
import numpy as np
from numpy.linalg import linalg, LinAlgError
from scipy.stats import chi2

__author__ = 'kcx'


class GaussianQuadraticTest:
    def __init__(self, grad_log_prob, scaling=2.0):
        self.scaling = scaling
        self.grad = grad_log_prob

    def k(self,x,y):
        return np.exp(-(x-y)**2/self.scaling)

    def g1k(self,x,y):
        return -2.0/self.scaling *self.k(x,y)*(x-y)

    def g2k(self,x,y):
        return -self.g1k(x,y)

    def gk(self,x,y):
        return 2.0 *self.k(x,y) *( self.scaling - 2*(x-y)**2)/self.scaling**2


    def compute_pvalue(self, samples,boots=100):

        N = samples.shape[0]
        U_matrix = np.zeros((N,N))
        for i in range(N):
            for j in range(N):

                x1 = samples[i]
                x2 = samples[j]
                a = self.grad(x1)*self.grad(x2)*self.k(x1,x2)
                b = self.grad(x2)*self.g1k(x1,x2)
                c = self.g1k(x1,x2)*self.g2k(x1,x2)
                d = self.gk(x1,x2)
                U_matrix[i,j] = a+b+c+d

        bootsraped_stats = np.zeros(boots)

        for proc in range(100):
            W = np.sign(np.random.randn(N))
            WW = np.outer(W,W)
            st = np.mean(U_matrix*WW)
            bootsraped_stats[proc] =st

        stat = np.mean(U_matrix)
        return  float(np.sum(bootsraped_stats > stat))/boots