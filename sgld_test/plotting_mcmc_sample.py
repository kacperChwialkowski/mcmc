from statsmodels.tsa.stattools import acf
from sgld_test.test import gen_X, log_probability
import seaborn as sns;
from sampplers.MetropolisHastings import metropolis_hastings
import numpy as np
sns.set(color_codes=True)
__author__ = 'kcx'


X = gen_X(400)

def vectorized_log_density(theta):
     return log_probability(theta,X)

sample = metropolis_hastings(vectorized_log_density, chain_size=1000, thinning=15, x_prev=np.random.randn(2))
# #
print(acf(sample[:,1],nlags=10))

with sns.axes_style("white"):
     sns.jointplot(x=sample[:,1], y=sample[:,0],kind='kde', color="k");
     sns.plt.show()
