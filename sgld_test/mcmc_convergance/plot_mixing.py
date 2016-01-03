from pandas import DataFrame
import seaborn

__author__ = 'kcx'
import  numpy as np

pvals = np.load('../pvals.npy')

n_rows, n_cols = pvals.shape

arr = np.empty((0,2))

for row_number in range(n_rows):
    for pval in pvals[row_number]:
        arr = np.vstack((arr, np.array([row_number,pval])))


df = DataFrame(arr)

seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()