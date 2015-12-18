import random
import math
import autograd.numpy as np
from autograd import grad


def log_normal(x):
    return  -(x)**2/2


grad_log_ugly = grad(log_normal)

def test_function(x,omega=0.4):
    return np.exp( -(x-omega)**2/2)

grad_test = grad(test_function)

def stat(x):
    return grad_log_ugly(x)*test_function(x) + grad_test(x)

N=10000
A = np.random.randn(N)



for i in range(N):
    A[i] = stat(A[i])

imnormal = (np.sqrt(N) * np.mean(A)) / np.var(A)
print(  abs(imnormal) < 4)



A = np.random.randn(N)*2

for i in range(N):
    A[i] = stat(A[i])

imnotnormal = (np.sqrt(N) * np.mean(A)) / np.var(A)
print(  abs(imnotnormal) > 4)