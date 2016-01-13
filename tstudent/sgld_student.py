__author__ = 'kcx'
from statsmodels.tsa.stattools import acf
from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn
import numpy as np


SGLD_EPSILON = 0.1

P_CHANGE = 0.1

N = 500

DEGREES_OF_FREEDOM = [1,3,5,7,9]+[1000]
MC_PVALUES_REPS = 500
TEST_CHAIN_SIZE = 10**6


# The null statistic is that random variables come form normal distibution, so the test statistic takes a gradient of
# logarithm of density of standard normal.
def grad_log_normal(x):
    return -x

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


# plain AR process
def normal_mild_corr(N):
    samples = np.zeros(N)
    X_t = 0
    a = 0.97
    innovation_var = np.sqrt(1 - a ** 2)
    for t in range(N):
        X_t = a*X_t + innovation_var *np.random.randn()
        samples[t] = X_t
    return samples

# estimate size of thinning
def get_thinning(X,nlags = 50):
    autocorrelation = acf(X, nlags=nlags, fft=True)
    # find correlation closest to 0.5
    thinning = np.argmin(np.abs(autocorrelation - 0.5)) + 1
    return thinning, autocorrelation[thinning]



X = sample_sgld_t_student(TEST_CHAIN_SIZE, 5.0, SGLD_EPSILON)
sgld_thinning, autocorr =  get_thinning(X)
print('thinning for sgld t-student simulation ', sgld_thinning)


X = normal_mild_corr(TEST_CHAIN_SIZE)
ar_thinning, autocorr =  get_thinning(X)
print('thinning for AR normal simulation ',ar_thinning)


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

    if mc % 15 == 0:
            print(' ',100.0*mc/MC_PVALUES_REPS,'%')
    X = normal_mild_corr(ar_thinning*N)
    X = X[::ar_thinning]
    pval =  get_pval(X,tester)
    results = np.vstack((results,np.array([np.Inf, pval])))



df = DataFrame(results)
pr =seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()

fig = pr.get_figure()
fig.savefig('../write_up/img/sgld_student.pdf')



