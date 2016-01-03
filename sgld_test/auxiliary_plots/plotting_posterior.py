import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;

from sgld_test.test import gen_X, _log_probability

sns.set(color_codes=True)

N = 1000
X = gen_X(N)
theta1 = np.arange(-2, 2, 0.25)
grid_size = len(theta1)
theta2 = np.arange(-2, 2, 0.25)
theta1, theta2 = np.meshgrid(theta1, theta2)
Z = np.copy(theta1)

log_den = np.exp(-(X ** 2 / 24.0)) / (4.0 * np.sqrt(6.0 * np.pi)) + np.exp(-(X ** 2 / 26)) / (2 * np.sqrt(26 * np.pi))

log_den = np.sum(np.log(log_den))

for i in range(grid_size):
    for j in range(grid_size):
        probability = _log_probability(theta1[i, j], theta2[i, j], X)
        Z[i, j] = np.exp(probability - log_den)

plt.figure()
CS = plt.contour(theta1, theta2, Z, 10)
plt.show()
