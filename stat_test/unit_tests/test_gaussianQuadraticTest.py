from unittest import TestCase
import numpy as np
from stat_test.Ustat import GaussianQuadraticTest

__author__ = 'kcx'


class TestGaussianQuadraticTest(TestCase):
    def grad_log_normal(self,x):
        return  -x


    def test_on_one_dim_gaussian(self):
        np.random.seed(42)
        data = np.random.randn(100)
        me = GaussianQuadraticTest(self.grad_log_normal)
        assert me.compute_pvalue(data) > 0.05

    def test_on_one_dim_gaussian(self):
        np.random.seed(42)
        data = np.random.randn(100)*2.0
        me = GaussianQuadraticTest(self.grad_log_normal)
        pval = me.compute_pvalue(data)
        print(pval)
        assert pval < 0.05
