import numpy as np

def metropolis_hastings(log_ugly,N = 10000,thining = 15, x_prev=np.random.randn()):
    A = [x_prev]
    for i in range(N*thining-1):
        guess = 2*np.random.randn()+x_prev
        old_log_lik = log_ugly(x_prev)
        new_log_lik = log_ugly(guess)
        if new_log_lik > old_log_lik:
            A.append(guess)
        else:
            u = np.random.uniform(0.0,1.0)
            if u < np.exp(new_log_lik - old_log_lik):
                A.append(guess)
            else:
                A.append(x_prev)
        x_prev = A[-1]
    return A[::thining]
