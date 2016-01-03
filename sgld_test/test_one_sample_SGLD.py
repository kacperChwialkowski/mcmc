from unittest import TestCase
import numpy as np
from sgld_test.bimodal_SGLD import one_sample_SGLD
from sgld_test.test import log_probability

__author__ = 'kcx'


class TestOne_sample_SGLD(TestCase):
    def test_one_sample_SGLD(self):
        np.random.seed(0)
        X = np.array([13.])
        def vectorized_log_density(theta):
            return log_probability(theta,X)

        # r = one_sample_SGLD(grad_the_log_density,grad_log_prior,X,n=1,chain_size=300,theta=np.array([0.,0.])))
