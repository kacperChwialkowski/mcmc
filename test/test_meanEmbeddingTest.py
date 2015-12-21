from unittest import TestCase
from test.stationary_distribution import MeanEmbeddingConsistanceTest
import numpy as np
__author__ = 'kcx'


class TestMeanEmbeddingTest(TestCase):
    def test_on_one_dim_gaus(self):
        np.random.seed(42)

        def log_normal(x):
            return  -(x)**2/2

        data = np.random.randn(10000)
        me = MeanEmbeddingConsistanceTest(data,log_normal)
        assert me.compute_pvalue()>0.05


    def test_on_four_dim_gaus(self):

        np.random.seed(42)
        def log_normal(x):
            return  -np.dot(x,x)/2

        data = np.random.randn(10000,4)
        me = MeanEmbeddingConsistanceTest(data,log_normal)
        assert me.compute_pvalue()>0.05