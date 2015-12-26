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

        def grad_log_normal(x):
            return  -x


        me = GaussianSteinTest(grad_log_normal,10)


        m = MeanEmbeddingConsistanceSelector(mh_gen, n=1000,thinning=15,tester=me)

        data,_ = m.points_from_stationary()

        me = GaussianSteinTest(grad_log_normal,10)
        assert me.compute_pvalue(data)>0.05

    def test_on_one_dim_gaus2(self):
        np.random.seed(42)
        k=2.0
        def grad_log_prob(x):

            return -(x/k + np.sin(x))*(1.0/k + np.cos(x))

        def log_prob(x):
            return -(x/5.0 + np.sin(x) )**2.0/2.0

        mh_gen = mh_generator(log_density=log_prob,x_start=1.0)
        me = GaussianSteinTest(grad_log_prob,41)


        m = MeanEmbeddingConsistanceSelector(mh_gen, n=1000,thinning=20,tester=me,max_ite=5)

        data,premature_stop = m.points_from_stationary()


        assert premature_stop is True
