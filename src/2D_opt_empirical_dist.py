import numpy as np
from matplotlib import pyplot as plt
import ot
import ot.plot



# generate data
n = 50
mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

xs = ot.datasets.get_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.get_2D_samples_gauss(n, mu_t, cov_t)


a, b = np.ones((n,)) / n, np.ones((n,)) / n


# loss matrix

M = ot.dist(xs, xt)

M /= M.max()

# plot samples


print("xs.shape : ", xs.shape)
print("M.shape : ", M.shape)


plt.figure(1)
plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
plt.legend(loc=0)
plt.title("Source and Target dis")



plt.figure(2)
plt.imshow(M, cmap='jet',  interpolation='nearest')
plt.title('Cost matrix M')

#%% EMD

G0 = ot.emd(a, b, M)


print(G0)
plt.figure(3)
plt.imshow(G0, cmap='jet', interpolation='nearest')
plt.title('OT matrix G0')


plt.figure(4)
ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
plt.legend(loc=0)
plt.title('OT matrix with samples')

plt.show()