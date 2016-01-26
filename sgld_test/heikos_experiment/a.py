import numpy as np
from pandas import DataFrame

p_vals = np.load('pvals.npy')
evals  = np.load('no_evals.npy')/10**6


epsilon = np.linspace(0.001, 0.2,25)

import seaborn as sns
import matplotlib.pyplot as plt

x = epsilon
x = ["%.2f" % v for v in x]
x[0]="0.001"
epsilon=x

data_pvals = []

for i,r in enumerate(p_vals):
    for eval in r:
         data_pvals.append( [epsilon[i], eval])

data_pvals = DataFrame(data_pvals)

data_evals = []
for i,r in enumerate(evals):
    for eval in r:
        data_evals.append([epsilon[i], eval])


data_evals = DataFrame(data_evals)
#
# sns.boxplot(x=0,y=1,data = data_pvals)
# sns.plt.show()
#
# exit(0)

# Set up the matplotlib figure
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)



sns.boxplot(x=0,y=1,data = data_pvals, palette="BuGn_d", ax=ax1)
ax1.set_ylabel("p values")
ax1.set_xlabel("")

y2 = evals
# y2 = np.mean(evals,axis=1)
sns.barplot(x=0,y=1,data = data_evals,palette="RdBu_r", ax=ax2)


ax2.set_ylabel("Likelihood evaluations")

sns.despine(bottom=True)
# plt.setp(f.axes, yticks=[])
plt.tight_layout(h_pad=3)
ax2.set(xlabel='epsilon')

plt.show()

f.savefig('../../write_up/img/Heiko.pdf')