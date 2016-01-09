from unittest import TestCase
import numpy as np
from stat_test.quadratic_time import GaussianQuadraticTest

__author__ = 'kcx'


class TestGaussianQuadraticTest(TestCase):
    def grad_log_normal(self,x):
        return  -x


    def test_regression_1(self):
        np.random.seed(43)
        data = np.random.randn(100)
        me = GaussianQuadraticTest(self.grad_log_normal)
        pval = me.compute_pvalue(data)
        assert pval == 0.79

    def test_regression_2(self):
        np.random.seed(42)
        data = np.random.randn(100)*2.0
        me = GaussianQuadraticTest(self.grad_log_normal)
        pval = me.compute_pvalue(data)
        assert pval == 0.0

