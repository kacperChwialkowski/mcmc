from unittest import TestCase
from sgld_test.constants import SIGMA_x

from sgld_test.gradients_of_likelihood import manual_grad


__author__ = 'kcx'

from autograd import grad
from numpy.testing import assert_almost_equal
from sgld_test.likelihoods import _log_lik
import seaborn as sns;

sns.set(color_codes=True)
import autograd.numpy as np  # Thinly-wrapped version of Numpy




def scalar_log_lik(theta_1, theta_2, x):
    arg = (x - theta_1)
    lik1 = 1.0 / np.sqrt(2 * SIGMA_x ** 2 * np.pi) * np.exp(- np.dot(arg, arg) / (2 * SIGMA_x ** 2))
    arg = (x - theta_1 - theta_2)
    lik2 = 1.0 / np.sqrt(2 * SIGMA_x ** 2 * np.pi) * np.exp(- np.dot(arg, arg) / (2 * SIGMA_x ** 2))

    return np.log(0.5 * lik1 + 0.5 * lik2)


assert_almost_equal(scalar_log_lik(1.0, 2.0, 3.0), _log_lik(1.0, 2.0, 3.0))

grad_the_log_density_x = grad(scalar_log_lik, 0)
grad_the_log_density_y = grad(scalar_log_lik, 1)



class TestManualLikelihoods(TestCase):
    def test_my_autograd_code(self):
        assert_almost_equal(grad_the_log_density_x(1., 2., 3.), 0.268941, decimal=4)
        assert_almost_equal(grad_the_log_density_y(1., 3., 3.), -0.339589, decimal=4)

    def test_manual_gradient(self):
        assert_almost_equal(grad_the_log_density_x(1., 3., 3.), manual_grad(1., 3., 3.)[0])
        assert_almost_equal(grad_the_log_density_y(1., 3., 3.), manual_grad(1., 3., 3.)[1])

    def test_vector_version(self):
        assert_almost_equal(manual_grad(1., 3., np.array([1.0, 3.0, 5.0]))[-1], manual_grad(1., 3., 5.))
        assert_almost_equal(manual_grad(1., 3., np.array([1.0, 3.0, 5.0]))[0], manual_grad(1., 3., 1.))
