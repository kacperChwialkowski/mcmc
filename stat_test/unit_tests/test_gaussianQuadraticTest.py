from nose.tools import assert_almost_equal
from unittest import TestCase

import numpy as np
from stat_test.quadratic_time import GaussianQuadraticTest
from numpy.testing.utils import assert_allclose


__author__ = 'kcx, heiko'


class TestGaussianQuadraticTest(TestCase):
    def grad_log_normal(self, x):
        return -x


    def test_regression_1(self):
        np.random.seed(43)
        data = np.random.randn(100)
        me = GaussianQuadraticTest(self.grad_log_normal)
        pval = me.compute_pvalue(data)
        assert pval == 0.79

    def test_regression_2(self):
        np.random.seed(42)
        data = np.random.randn(100) * 2.0
        me = GaussianQuadraticTest(self.grad_log_normal)
        pval = me.compute_pvalue(data)
        assert pval == 0.0

    def test_k_multiple_equals_k_no_grad_multiple_given(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        K = me.k_multiple(X)
        
        for i in range(N):
            for j in range(N):
                k = me.k(X[i], X[j])
                assert_almost_equal(K[i, j], k)
    
    def test_k_multiple_equals_k_grad_multiple_given(self):
        def fun(self, X):
            return -X
        
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal, grad_log_prob_multiple=fun)
        K = me.k_multiple(X)
        
        for i in range(N):
            for j in range(N):
                k = me.k(X[i], X[j])
                assert_almost_equal(K[i, j], k)

    def test_g1k_multiple_equals_g1k(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        G1K = me.g1k_multiple(X)
         
        for i in range(N):
            for j in range(N):
                g1k = me.g1k(X[i], X[j])
                assert_almost_equal(G1K[i, j], g1k)
    
    def test_g2k_multiple_equals_g2k(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        G2K = me.g2k_multiple(X)
         
        for i in range(N):
            for j in range(N):
                g2k = me.g2k(X[i], X[j])
                assert_almost_equal(G2K[i, j], g2k)

    def test_gk_multiple_equals_gk(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        GK = me.gk_multiple(X)
         
        for i in range(N):
            for j in range(N):
                gk = me.gk(X[i], X[j])
                assert_almost_equal(GK[i, j], gk)
    
    def test_get_statistic_multiple_naive_linear_loop_equals_get_statistic(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        stat_multiple, U_matrix_multiple = me.get_statistic_multiple_naive_linear_loop(X)
        stat, U_matrix = me.get_statisitc(N, X)
        
        assert_allclose(stat, stat_multiple)
        assert_allclose(U_matrix_multiple, U_matrix)
    
    def test_get_statistic_multipleequals_get_statistic(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        stat_multiple, U_matrix_multiple = me.get_statistic_multiple(X)
        stat, U_matrix = me.get_statisitc(N, X)
        
        assert_allclose(stat, stat_multiple)
        assert_allclose(U_matrix_multiple, U_matrix)