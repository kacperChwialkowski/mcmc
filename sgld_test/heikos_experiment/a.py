import numpy as np

p_vals = np.load('pvals.npy')
evals  = np.load('no_evals.npy')

p_vals = np.min(p_vals,axis=0)

print(p_vals)
