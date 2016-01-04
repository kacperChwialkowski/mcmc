from unittest import TestCase
import numpy as np
from numpy.testing.utils import assert_almost_equal
from sgld_test.bimodal_SGLD import vSGLD
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior

__author__ = 'kcx'


class TestOne_sample_SGLD(TestCase):
    def test_one_sample_SGLD(self):
        np.random.seed(0)
        X = np.array([13.])
        r2 = vSGLD(manual_grad,grad_log_prior,X,n=1,chain_size=299,theta=np.array([1.,1.3]))
        res = np.array([ 2.16361105,  6.51715066])
        assert_almost_equal(r2[-1],res)


    def test_one_sample_SGLD(self):
        np.random.seed(0)
        X = np.array([13.])
        r2 = vSGLD(manual_grad,grad_log_prior,X,n=1,chain_size=29,theta=np.array([1.,1.3]))
        res = np.array(  [ 1.60282887,1.95290384])
        assert_almost_equal(r2[-1],res)