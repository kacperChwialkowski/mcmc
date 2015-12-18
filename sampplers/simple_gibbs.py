import random
import math
import autograd.numpy as np
from autograd import grad


def log_ugly(x):
    return -(x/10 + np.sin(x) )**2/2


grad_log_ugly = grad(log_ugly)

def test_function(x,omega=0.4):
    return np.exp( -(x-omega)**2/2)

grad_test = grad(test_function)

def stat(x):
    return grad_log_ugly(x)*test_function(x) + grad_test(x)

x_prev = 0.1




A = [x_prev]
N = 100000
for i in range(N-1):
    guess = np.random.randn()+x_prev
    old_log_lik = log_ugly(x_prev)
    new_log_lik = log_ugly(guess)
    if new_log_lik > old_log_lik:
        A.append(guess)
    else:
        u = random.uniform(0.0,1.0)
        if (u < math.exp(new_log_lik - old_log_lik)):
            A.append(guess)
        else:
            A.append(x_prev)
    x_prev = A[-1]

import matplotlib.pyplot as plt



assert len(A)==N


for i in range(N):
    A[i] = stat(A[i])

rows = int(N/1000)

A = np.reshape(A,(rows,1000))

imnormal = np.zeros(rows)
for row in range(rows):
    imnormal[row] = np.abs((np.sqrt(1000.0) * np.mean(A[row])) / np.var(A[row]))



plt.plot(imnormal)

plt.show()