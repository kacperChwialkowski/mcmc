import numpy as np


#   THIS IS ULTRA SPECIFIC TO THE  PROBLEM, Dont dare to use it!!!!
def SGLD(log_density,grad_log_density,grad_log_prior, X,n,chain_size=10000, thinning=15, x_prev=np.random.randn(2)):
    A = [x_prev]
    N = X.shape[0]

    for t in range(chain_size*thinning-1):
        print(t)

        gamma = -0.55
        epsilon_t = 0.1*(1+t)**gamma

        noise = np.sqrt(epsilon_t)*np.random.randn()

        sub = np.random.choice(X, n)

        stupid_sum=np.array([0.0,0.0])
        for data_point in sub:
            stupid_sum = stupid_sum+ grad_log_density(x_prev,data_point)

        grad = grad_log_prior(x_prev) + (N/n)*stupid_sum

        grad = grad*epsilon_t/2
        print(grad,noise,epsilon_t)

        guess = grad+noise

        old_log_lik = log_density(x_prev)
        new_log_lik = log_density(guess)
        if new_log_lik > old_log_lik:
            A.append(guess)
        else:
            u = np.random.uniform(0.0,1.0)
            if u < np.exp(new_log_lik - old_log_lik):
                A.append(guess)
            else:
                A.append(x_prev)
        x_prev = A[-1]
    return np.array(A[::thinning])




class sgld_generator:

    def __init__(self,log_density,x_start=np.random.randn()):
        self.log_density = log_density
        self.x_last = x_start

    def get(self, chunk_size,thinning):
        data = metropolis_hastings(self.log_density,chain_size=chunk_size,thinning=thinning,x_prev=self.x_last)
        self.x_last = data[-1]
        return data