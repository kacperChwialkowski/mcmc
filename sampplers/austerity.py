from nose.tools import assert_almost_equal
from scipy.stats import t
from sgld_test.likelihoods import log_probability

__author__ = 'kcx'


import numpy as np

def austerity(log_lik,log_density_prior, X,epsilon,batch_size=30,chain_size=10000, thinning=15, theta_t=np.random.randn()):
    A = [theta_t]
    N  = X.shape[0]
    dimension=1
    if hasattr(theta_t, "__len__"):
        dimension = len(theta_t)

    for i in range(chain_size*thinning-1):

        theta_prime = np.random.randn(dimension)+theta_t

        u = np.random.rand()
        mu_0 = np.log(u)+log_density_prior(theta_t) -log_density_prior(theta_prime)
        mu_0 = mu_0/N

        # sub = log_lik(X, theta_prime) - log_lik(X, theta_t)
        # l_hat = np.sum(sub)+ log_density_prior(theta_prime) - log_density_prior(theta_t)
        #
        # fuck_me = log_probability(theta_prime, X ) - log_probability(theta_t, X )
        # print(l_hat,fuck_me)
        # print('fff',np.mean(sub))
        # assert_almost_equal( l_hat ,fuck_me)

        accept = approximate_MH_accept(mu_0, log_lik, X, batch_size, epsilon, theta_prime, theta_t, N)
        if accept:
           theta_t = theta_prime

        A.append(theta_t)

        #
        # if np.log(u) <fuck_me:
        #     aux_accept = True
        # else:
        #     aux_accept = False
        #
        # huj = mu_0 <  np.mean(sub)
        # print(huj,accept)
        # assert huj == aux_accept
        # assert aux_accept == accept


    return np.array(A[::thinning])


def approximate_MH_accept(mu_0,log_lik,X,batch_size,epsilon,theta_prime, theta_t,N):

    iteration_number=0

    while True:
        iteration_number +=1
        n = iteration_number*batch_size
        n = min(n, N)
        sub = np.random.choice(X, n,replace=False)
        sub = log_lik(sub, theta_prime) - log_lik(sub, theta_t)
        l_hat = np.mean(sub)
        l_2_hat = np.mean(sub**2)
        s_l = np.sqrt(l_2_hat - l_hat**2*n/(n-1))
        s = s_l/ np.sqrt(n)*np.sqrt(1 - (n-1)/(N-1))
        t_students_var = (l_hat - mu_0) / s
        stat = np.abs(t_students_var)
        delta  = t.sf(stat, n-1)
        if delta < epsilon:
            if l_hat > mu_0:
                return True

            return False
