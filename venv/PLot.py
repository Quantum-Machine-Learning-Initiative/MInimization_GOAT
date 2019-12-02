# libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# the function that I'm going to plot


def f(theta, phi):
    return np.power(np.cos(theta), 2)


def g(theta, phi):
    return 1/2 + 0.5*np.sin(2*theta)


theta = np.arange(0, 2*np.pi, 0.1)
phi = np.arange(0, 2*np.pi, 0.1)
Theta, Phi = np.meshgrid(theta, phi)
F = f(Theta, Phi)
G = g(Theta, Phi)
# im = plt.imshow(F, cmap=matplotlib.cm.RdBu)
# matplotlib.axes.Axes.set_xlim(right=0, left=6)

ax1 = plt.subplot(211)
plt.imshow(F)
ax2 = plt.subplot(212)
plt.imshow(G)

# create data
# x = np.random.poisson(size=50000)
# g = np.random.normal(size=50000)
# y = x * 3 + np.random.normal(size=50000)

# Big bins
# plt.hist2d(f, g, bins=(50, 50), cmap=plt.cm.jet)
# plt.hist2d(f, g, bins=(50, 50))
# plt.show()

# Small bins
# plt.hist2d(x, y, bins=(300, 300), cmap=plt.cm.jet)
# plt.show()

# If you do not set the same values for X and Y, the bins aren't square !
# plt.hist2d(x, y, bins=(300, 30), cmap=plt.cm.jet)

# plt.show()

plt.show()
