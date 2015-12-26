import warnings
import autograd.numpy as np
from two_sample_test.utils import mahalanobis_distance

__author__ = 'kcx'


class GaussianSteinTest:

    def __init__(self, grad_log_prob,num_random_freq):
        self.number_of_random_frequencies = num_random_freq

        def stein_stat(random_frequency,samples):
            a = grad_log_prob(samples)
            b = self._gaussian_test_function(samples, random_frequency)
            c = self._test_function_grad(samples, random_frequency)
            return a * b + c

        self.stein_stat = stein_stat


    def _make_two_dimensional(self, z):
        if len(z.shape) == 1:
            z = z[:, np.newaxis]
        return z

    def _get_mean_embedding(self, random_frequency, x,scaling =2.0):
        z = x - random_frequency
        z = np.linalg.norm(z, axis=1) ** 2
        z = np.exp(-z / scaling)
        return z

    def _gaussian_test_function(self,x,random_frequency,scaling =2.0):
        x = self._make_two_dimensional(x)
        mean_embedding = self._get_mean_embedding(random_frequency, x,scaling)
        return np.tile(mean_embedding,(self.shape,1)).T


    def _test_function_grad(self,x,omega,scaling = 2.0):
        arg = (x - omega)*2.0/scaling
        test_function_val = self._gaussian_test_function(x, omega,scaling)
        return -arg* test_function_val



    def compute_pvalue(self, samples):

        samples = self._make_two_dimensional(samples)

        self.shape = samples.shape[1]

        stats_for_freqs = []

        for f in range(self.number_of_random_frequencies):
            matrix_of_stats = self.stein_stat(random_frequency=np.random.randn(),samples=samples)
            stats_for_freqs.append(matrix_of_stats)

        normal = np.hstack(stats_for_freqs)
        normal = self._make_two_dimensional(normal)

        return mahalanobis_distance(normal,normal.shape[1])


class MeanEmbeddingConsistanceSelector:



    def __init__(self, data_generator, n,thinning, tester,alpha=0.05 ,max_ite=10):
        self.data_generator = data_generator
        self.thinning = thinning
        self.tester = tester
        self.n=n
        self.alpha = alpha
        self.max_ite = max_ite

    def points_from_stationary(self):
        run = True
        zeta2 = 1.645
        data = self.data_generator.get(self.n,self.thinning)

        indicator = 1.0
        level = 1/(indicator**2)*(1/zeta2)*self.alpha

        while self.tester.compute_pvalue(data) < level and run:
            data = self.data_generator.get(self.n,self.thinning)

            indicator = indicator+1
            level = 1/(indicator**2)*(1/zeta2)*self.alpha
            run = indicator < self.max_ite
            print(indicator,self.max_ite)

        premature_stop = not run
        if premature_stop:
            warnings.warn('didnt converge')

        return data,premature_stop


