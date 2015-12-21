from multiprocessing.pool import Pool
import autograd.numpy as np
from autograd import grad
from two_sample_test.utils import mahalanobis_distance

__author__ = 'kcx'


class MeanEmbeddingConsistanceTest:


    def test_function(self,omega):
        def f(x):
            arg = (x - omega)
            return np.exp( -np.dot(arg,arg)/2.0)
        return f

    def __init__(self, data_x, log_probability, scale=1,freq=np.random.randn()):
        self.data_x = scale*data_x
        self.log_probability = log_probability

        self.scale = scale

        self.grad_log = grad(log_probability)
        self.test_function = self.test_function(freq)
        self.grad_test = grad(self.test_function)




    def stat(self, x):
        grad_log = self.grad_log(x)
        grad_test = self.grad_test(x)
        return grad_log *self.test_function(x) + grad_test


    def compute_pvalue(self):
        # pool = Pool(processes=4)
        normal = np.array([self.stat(x) for x in self.data_x])
        if len(normal.shape)==1:
            normal = normal[:,np.newaxis]

        return mahalanobis_distance(normal,normal.shape[1])



np.random.seed(42)

def log_normal(x):
    return  -np.dot(x,x)/2


# data = np.random.randn(10000)
# me = MeanEmbeddingConsistanceTest(data,log_normal)
# assert me.compute_pvalue()>0.05

data = np.random.randn(10000,4)
me = MeanEmbeddingConsistanceTest(data,log_normal)
assert me.compute_pvalue()>0.05
