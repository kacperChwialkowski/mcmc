import warnings
import autograd.numpy as np
from two_sample_test.utils import mahalanobis_distance

__author__ = 'kcx'


class GaussianSteinTest:

    def __init__(self, grad_log_prob,num_random_freq):
        self.number_of_random_frequencies = num_random_freq

        def stein_stat(random_frequency,samples):
            a = grad_log_prob(samples)
            b = self.gaussian_test_function(samples, random_frequency)
            c = self.test_function_grad(samples, random_frequency)
            return a * b + c

        self.stein_stat = stein_stat


    def make_two_dimensional(self, z):
        if len(z.shape) == 1:
            z = z[:, np.newaxis]
        return z

    def gaussian_test_function(self,x,random_frequency):
        z = x - random_frequency

        z = self.make_two_dimensional(z)

        z2 = np.linalg.norm(z, axis=1)**2
        z2= np.exp(-z2/2.0)
        return np.tile(z2,(self.shape,1)).T


    def test_function_grad(self,x,omega):
        arg = (x - omega)
        test_function_val = self.gaussian_test_function(x, omega)
        return -arg* test_function_val



    def compute_pvalue(self, sampless):

        sampless = self.make_two_dimensional(sampless)

        self.shape = sampless.shape[1]

        stats_for_freqs = []

        for f in range(self.number_of_random_frequencies):
            matrix_of_stats = self.stein_stat(random_frequency=np.random.randn(),samples=sampless)
            stats_for_freqs.append(matrix_of_stats)

        normal = np.hstack(stats_for_freqs)
        normal = self.make_two_dimensional(normal)

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


