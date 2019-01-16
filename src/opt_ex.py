import numpy as np
from matplotlib import pyplot as plt

import ot
import ot.plot
from ot.datasets import get_1D_gauss as gauss

#%% parameters

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = gauss(n, m=20, s=5)  # m= mean, s= std
b = gauss(n, m=60, s=10)

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()

#%% plot the distributions

plt.figure(1, figsize=(6.4, 3))
plt.plot(x, a, 'b', label='Source distribution')
plt.plot(x, b, 'r', label='Target distribution')
plt.legend()

#%% plot distributions and loss matrix

plt.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, M, 'Cost matrix M')





# solve EMD

#%% EMD

G0 = ot.emd(a, b, M)

plt.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')









plt.show()