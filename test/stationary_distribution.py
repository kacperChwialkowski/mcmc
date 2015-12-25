import warnings
import autograd.numpy as np
from two_sample_test.utils import mahalanobis_distance

__author__ = 'kcx'


class GaussianSteinTest:

    def __init__(self, samples, grad_log_prob,num_random_freq):
        self.num_random_freq = num_random_freq
        self.shape=1

        if len(samples.shape)==1:
            samples = samples[:,np.newaxis]

        self.shape = samples.shape[1]

        def statf(freq):

            a = grad_log_prob(samples)
            b = self.test_function(samples, freq)
            c = self.test_function_grad(samples, freq)
            return a * b + c

        self.statf = statf


    def test_function(self,x,omega):
        z = x - omega
        if len(z.shape)==1:
            z = z[:,np.newaxis]

        z2 = np.linalg.norm(z, axis=1)**2
        z2= np.exp(-z2/2.0)
        return np.tile(z2,(self.shape,1)).T


    def test_function_grad(self,x,omega):
        arg = (x - omega)
        test_function_val = self.test_function(x, omega)
        return -arg* test_function_val



    def compute_pvalue(self):

        stats_for_freqs = []
        for f in range(self.num_random_freq):
            matrix_of_stats = self.statf(freq=np.random.randn())
            stats_for_freqs.append(matrix_of_stats)

        normal = np.hstack(stats_for_freqs)
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
        me = GaussianSteinTest(data,self.log_probability,self.scale,self.freq)
        print('lame',indicator)
        while me.compute_pvalue() < level or stop:
            print('lame',indicator)
            data = self.data_generator.get(self.n,self.thinning)

            indicator = indicator+1
            level = 1/(indicator**2)*(1/zeta2)*self.alpha
            me = GaussianSteinTest(data,self.log_probability,self.scale,self.freq)
            stop = indicator > self.max_ite
        if stop:
            warnings.warn('didnt converge')
        return data


