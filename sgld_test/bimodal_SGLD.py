import numpy as np


#   THIS IS ULTRA SPECIFIC TO THE  PROBLEM, Dont dare to use it!!!!
TRUE_B = 2.3101


def SGLD(grad_log_density,grad_log_prior, X,n,log_density,chain_size=10000, thinning=1, x_prev=np.random.rand(2),epsilon=5*10.0**(-3)):
    Accpetance = []
    Samples = [x_prev]
    N = X.shape[0]
    old_log_lik  = log_density(x_prev)
    for t in range(chain_size*thinning-1):

        epsilon_t = epsilon

        noise = np.sqrt(epsilon_t)*np.random.randn(2)

        sub = np.random.choice(X, n)

        stupid_sum=np.array([0.0,0.0])
        for data_point in sub:
            stupid_sum = stupid_sum+ grad_log_density(x_prev,data_point)

        grad = grad_log_prior(x_prev) + (N/n)*stupid_sum

        grad = grad*epsilon_t/2


        x_prev = x_prev+grad+noise
        Samples.append(x_prev)
        #
        new_log_lik = log_density(x_prev)
        if new_log_lik > old_log_lik:
            Accpetance.append(1.0)
            old_log_lik = new_log_lik
        else:
            u = np.random.uniform(0.0,1.0)
            if u < np.exp(new_log_lik - old_log_lik):
                Accpetance.append(1)
                old_log_lik = new_log_lik
            else:
                Accpetance.append(0)

    return np.array(Samples[::thinning]),Accpetance

