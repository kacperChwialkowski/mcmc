from multiprocessing.pool import Pool
import warnings
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


class MeanEmbeddingConsistanceSelector:



    def __init__(self, data_generator, n,thinning,log_probability,alpha=0.05, scale=1,freq=np.random.randn(),max_ite=100):
        self.data_generator = data_generator
        self.thinning = thinning

        self.log_probability = log_probability

        self.scale = scale
        self.n=n
        self.alpha = alpha
        self.freq = freq
        self.max_ite = max_ite

    def points_from_stationary(self):
        stop = False
        zeta2 = 1.645
        data = self.data_generator.get(self.n,self.thinning)

        indicator = 1.0
        level = 1/(indicator**2)*(1/zeta2)*self.alpha
        me = MeanEmbeddingConsistanceTest(data,self.log_probability,self.scale,self.freq)
        print('lame',indicator)
        while me.compute_pvalue() < level or stop:
            print('lame',indicator)
            data = self.data_generator.get(self.n,self.thinning)

            indicator = indicator+1
            level = 1/(indicator**2)*(1/zeta2)*self.alpha
            me = MeanEmbeddingConsistanceTest(data,self.log_probability,self.scale,self.freq)
            stop = indicator > self.max_ite
        if stop:
            warnings.warn('didnt converge')
        return data


