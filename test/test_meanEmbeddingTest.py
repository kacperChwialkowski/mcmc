from unittest import TestCase
from test.stationary_distribution import GaussianSteinTest
import numpy as np
__author__ = 'kcx'


class TestMeanEmbeddingTest(TestCase):


    def test_on_one_dim_gaus(self):
        np.random.seed(42)

        def grad_log_normal(x):
            return  -x


        data = np.random.randn(10000)
        me = GaussianSteinTest(grad_log_normal,1)
        assert me.compute_pvalue(data)>0.05


    def test_on_four_dim_gaus(self):

        np.random.seed(42)
        def grad_log_normal(x):
            return  -x

        data = np.random.randn(10000,4)
        me = GaussianSteinTest(grad_log_normal,1)
        assert me.compute_pvalue(data) > 0.05

    def test_on_one_dim_gaus2(self):
        np.random.seed(42)

        def grad_log_normal(x):
            return  -x


        data = np.random.randn(10000)
        me = GaussianSteinTest(grad_log_normal,3)
        assert me.compute_pvalue(data)>0.05


    def test_on_four_dim_gaus2(self):

        np.random.seed(42)
        def grad_log_normal(x):
            return  -x

        data = np.random.randn(10000,4)
        me = GaussianSteinTest(grad_log_normal,3)
        assert me.compute_pvalue(data) > 0.05


    def test_on_four_dim_gaus2(self):

        np.random.seed(42)
        def grad_log_normal(x):
            return  -x

        data = np.random.randn(10000,4)+0.01*np.random.rand()
        me = GaussianSteinTest(grad_log_normal,10)
        p1 = me.compute_pvalue(data)
        me = GaussianSteinTest(grad_log_normal,1)
        p2 = me.compute_pvalue(data)
        print(p1,p2)
        assert p1 <p2
