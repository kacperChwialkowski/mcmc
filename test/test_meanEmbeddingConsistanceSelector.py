from unittest import TestCase
from sampplers.MetropolisHastings import mh_generator
from test.stationary_distribution import MeanEmbeddingConsistanceSelector, GaussianSteinTest
import autograd.numpy as np

__author__ = 'kcx'


class TestMeanEmbeddingConsistanceSelector(TestCase):


    def test_on_one_dim_gaus(self):
        np.random.seed(42)
        def log_normal(x):
                return  -np.dot(x,x)/2
        mh_gen = mh_generator(log_density=log_normal)

        m = MeanEmbeddingConsistanceSelector(mh_gen, n=10000,thinning=15, log_probability=log_normal)

        data = m.points_from_stationary()

        me = GaussianSteinTest(data,log_normal)
        assert me.compute_pvalue()>0.05

    def test_on_one_dim_gaus2(self):
        np.random.seed(42)
        def log_ugly(x):
            return -(x/20.0 + np.sin(x) )**2.0/2.0

        def log_ugly_fake(x):
            return -(x/10.0 + np.sin(x) )**2.0/2.0

        mh_gen = mh_generator(log_density=log_ugly_fake,x_start=100.0)
        m = MeanEmbeddingConsistanceSelector(mh_gen, n=15*1000,thinning=15, log_probability=log_ugly)

        data = m.points_from_stationary()

        me = GaussianSteinTest(data,log_ugly)
        assert me.compute_pvalue()>0.05
