from sampplers.MetropolisHastings import metropolis_hastings

__author__ = 'kcx'
from statsmodels.tsa.stattools import acf
from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn
import numpy as np
from tools.latex_plot_init import plt

SGLD_EPSILON = 0.0478

P_CHANGE = 0.03

N = 650

DEGREES_OF_FREEDOM = []
MC_PVALUES_REPS = 100
TEST_CHAIN_SIZE = 2*10**6


# The null statistic is that random variables come form normal distibution, so the test statistic takes a gradient of
# logarithm of density of standard normal.
def grad_log_normal(x):
    return -x

def log_normal(x):
    return -x**2.0/2.0

# sampling approximately from t student in sgld style
def sample_sgld_t_student(N,degree_of_freedom,epsilon):
    samples = np.zeros(N)
    X_t = 0
    for t in range(N):
        grad_log_tstudent = (-(1 + degree_of_freedom) * X_t / (degree_of_freedom + X_t ** 2.0))
        delta = epsilon/2.0* grad_log_tstudent + np.sqrt(epsilon)*np.random.randn()
        X_t = X_t + delta
        samples[t] = X_t
    return samples


def sample_sgld_t_student(N,degree_of_freedom,epsilon):
    X =  np.random.standard_t(degree_of_freedom,N)
    for i in range(2,N):
        if np.random.rand()>epsilon:
            X[i] = X[i-np.random.randint(1,np.min((i,5)))]
    return X

def grad_log_t_df(df):
    def grad_log_t(x):
        return -(df+1.0)/2.0*np.log(1+x**2/df)
    return grad_log_t

def sample_sgld_t_student(N,degree_of_freedom,epsilon):
    grd_log = grad_log_t_df(degree_of_freedom)
    X =  metropolis_hastings(grd_log, chain_size=N, thinning=1, x_prev=np.random.randn(), step=0.25)
    return X

# plain AR process
def normal_mild_corr(N):
    samples = np.zeros(N)
    X_t = 0
    a = 0.95
    innovation_var = np.sqrt(1 - a ** 2)
    for t in range(N):
        X_t = a*X_t + innovation_var *np.random.randn()
        samples[t] = X_t
    return samples

# plain AR process
def normal_mild_corr(N):
    X =  np.random.randn(N)
    for i in range(2,N):
        if np.random.rand() > SGLD_EPSILON:
            X[i] = X[i-np.random.randint(1,np.min((i,5)))]
    return X

def normal_mild_corr(N):
    X =  metropolis_hastings(log_normal, chain_size=N, thinning=1, x_prev=np.random.randn(),step=0.5)
    return X



# estimate size of thinning
def get_thinning(X,nlags = 50):
    autocorrelation = acf(X, nlags=nlags,fft=True)
    # find correlation closest to given v
    thinning = np.argmin(np.abs(autocorrelation - 0.6)) + 1
    return thinning, autocorrelation

X = normal_mild_corr(TEST_CHAIN_SIZE)
ar_thinning, autocorr =  get_thinning(X)
print('thinning for AR normal simulation ',ar_thinning,autocorr[ar_thinning])



X = sample_sgld_t_student(TEST_CHAIN_SIZE, 100.0, SGLD_EPSILON)
sgld_thinning, autocorr =  get_thinning(X)
print('thinning for sgld t-student simulation ', sgld_thinning,autocorr[sgld_thinning])


# plt.plot(np.log(autocorr))
# plt.show()



results = np.empty((0,2))



tester = GaussianQuadraticTest(grad_log_normal)

def get_pval(X,tester):
    U_stat, _ = tester.get_statistic_multiple(X)
    return tester.compute_pvalues_for_processes(U_stat, P_CHANGE)

for df in DEGREES_OF_FREEDOM:
    print(df)
    for mc in range(MC_PVALUES_REPS):
        if mc % 32 == 0:
            print(' ',100.0*mc/MC_PVALUES_REPS,'%')
        X = sample_sgld_t_student(sgld_thinning *N,df, SGLD_EPSILON)
        X = X[::sgld_thinning]
        pval = get_pval(X,tester)
        results = np.vstack((results,np.array([df, pval])))


print('Inf')
for mc in range(MC_PVALUES_REPS):

    if mc % 32 == 0:
            print(' ',100.0*mc/MC_PVALUES_REPS,'%')
    X = normal_mild_corr(ar_thinning*N)
    X = X[::ar_thinning]
    pval =  get_pval(X,tester)
    results = np.vstack((results,np.array([np.Inf, pval])))

np.save('results.npy',results)





