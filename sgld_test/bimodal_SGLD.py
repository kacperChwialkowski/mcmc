import numpy as np


#   THIS IS ULTRA SPECIFIC TO THE  PROBLEM, Dont dare to use it!!!!
TRUE_B = 2.3101


def SGLD(grad_log_density,grad_log_prior, X,n,chain_size=10000, thinning=1, x_prev=np.random.rand(2),epsilon=5*10.0**(-3)):
    Samples = [x_prev]
    N = X.shape[0]

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


    return np.array(Samples[::thinning])





def vSGLD(grad_log_density,grad_log_prior, X,n,chain_size=10000,  theta=3*np.random.randn(2),epsilon=5*10.0**(-3)):
    N = X.shape[0]
    X = np.random.permutation(X)

    samples = np.zeros((chain_size,2))
    for t in range(chain_size):

        epsilon_t = epsilon
        noise = np.sqrt(epsilon_t)*np.random.randn(2)

        sub = np.random.choice(X, n)

        stupid_sum = np.sum(grad_log_density(theta[0],theta[1],sub),axis=0)

        grad = grad_log_prior(theta) + (N/n)*stupid_sum

        grad = grad*epsilon_t/2

        theta = theta+grad+noise
        samples[t] = theta

    return samples


def evSGLD(grad_log_density,grad_log_prior, X,n,chain_size=10000,  theta=3*np.random.randn(2)):
    N = X.shape[0]
    X = np.random.permutation(X)

    samples = np.zeros((chain_size,2))
    for t in range(chain_size):

        b=2.31
        a = 0.01584 #originally it was 10 times smaller
        epsilon_t = a*(b+t)**(-0.55)

        noise = np.sqrt(epsilon_t)*np.random.randn(2)

        sub = np.random.choice(X, n)

        stupid_sum = np.sum(grad_log_density(theta[0],theta[1],sub),axis=0)

        grad = grad_log_prior(theta) + (N/n)*stupid_sum

        grad = grad*epsilon_t/2

        theta = theta+grad+noise
        samples[t] = theta

    return samples