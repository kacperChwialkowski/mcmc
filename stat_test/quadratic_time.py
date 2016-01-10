from scipy.spatial.distance import squareform, pdist

import numpy as np


__author__ = 'kcx, heiko'


class GaussianQuadraticTest:
    def __init__(self, grad_log_prob, scaling=2.0, grad_log_prob_multiple=None):
        self.scaling = scaling
        self.grad = grad_log_prob
        
        # construct (slow) multiple gradient handle if efficient one is not given
        if grad_log_prob_multiple is None:
            def grad_multiple(X):
                # simply loop over grad calls. Slow
                return np.array([self.grad(x) for x in X])
            
            self.grad_multiple = grad_multiple
        else:
            self.grad_multiple = grad_log_prob_multiple
            
    def k(self, x, y):
        return np.exp(-(x - y) ** 2 / self.scaling)
    
    def k_multiple(self, X):
        """
        Efficient computation of kernel matrix without loops
        
        Effectively does the same as calling self.k on all pairs of the input
        """
        assert(X.ndim == 1)
        
        sq_dists = squareform(pdist(X.reshape(len(X), 1), 'sqeuclidean'))
            
        K = np.exp(-(sq_dists) / self.scaling)
        return K

    def g1k(self, x, y):
        return -2.0 / self.scaling * self.k(x, y) * (x - y)
    
    def g1k_multiple(self, X):
        """
        Efficient gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.g1k on all pairs of the input
        """
        assert X.ndim == 1
        
        differences = X.reshape(len(X), 1) - X.reshape(1, len(X))
        sq_differences = differences ** 2
        K = np.exp(-sq_differences / self.scaling)

        return -2.0 / self.scaling * K * differences

    def g2k(self, x, y):
        return -self.g1k(x, y)
    
    def g2k_multiple(self, X):
        """
        Efficient 2nd gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.g2k on all pairs of the input
        """
        return -self.g1k_multiple(X)

    def gk(self, x, y):
        return 2.0 * self.k(x, y) * (self.scaling - 2 * (x - y) ** 2) / self.scaling ** 2

    def gk_multiple(self, X):
        """
        Efficient gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.gk on all pairs of the input
        """
        assert X.ndim == 1
        
        differences = X.reshape(len(X), 1) - X.reshape(1, len(X))
        sq_differences = differences ** 2
        K = np.exp(-sq_differences / self.scaling)

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2

    def get_statisitc(self, N, samples):
        U_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                x1 = samples[i]
                x2 = samples[j]
                a = self.grad(x1) * self.grad(x2) * self.k(x1, x2)
                b = self.grad(x2) * self.g1k(x1, x2)
                c = self.grad(x1) * self.g2k(x1, x2)
                d = self.gk(x1, x2)
                U_matrix[i, j] = a + b + c + d
        stat = N * np.mean(U_matrix)
        return U_matrix, stat
    
    def get_statistic_multiple(self, samples):
        """
        Efficient statistic computation with multiple inputs
        
        Effectively does the same as calling self.get_statisitc.
        """
        log_pdf_gradients = self.grad_multiple(samples)
        K = self.k_multiple(samples)
        G1K = self.g1k_multiple(samples)
        G2K = self.g2k_multiple(samples)
        GK = self.gk_multiple(samples)
        
        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U) 
        return U, stat

    def compute_pvalue(self, samples, boots=100):

        N = samples.shape[0]
        U_matrix, stat = self.get_statisitc(N, samples)


        bootsraped_stats = np.zeros(boots)

        for proc in range(100):
            W = np.sign(np.random.randn(N))
            WW = np.outer(W, W)
            st = np.mean(U_matrix * WW)
            bootsraped_stats[proc] = N * st

        return  float(np.sum(bootsraped_stats > stat)) / boots
