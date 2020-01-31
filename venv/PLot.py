# libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# the function that I'm going to plot


def f(_theta, _phi):
    return np.power(np.cos(_theta), 2)


def g(_theta, _phi):
    return (1/2)*(np.power(np.power(np.cos(_theta)+np.cos(_phi)*np.sin(_theta), 2) + np.power(np.sin(_phi) * np.sin(_theta), 2), 0.5))


def f2(_theta, _phi):
    x1 = np.array([0, 1])
    x2 = (np.power(1 / 2, 0.5)) * np.array([np.sin(_theta) * np.exp(1.j * _phi), np.cos(_theta)])
    return np.power(np.abs(x1.dot(x2)), 2)


def g2(_theta, _phi):
    x1 = np.array([np.power(1 / 2, 0.5), np.power(1 / 2, 0.5)])
    x2 = (np.power(1 / 2, 0.5)) * np.array([np.sin(_theta) * np.exp(1.j * _phi), np.cos(_theta)])
    return np.power(np.abs(x1.dot(x2)), 2)


def g2_i(_theta, _phi):
    x1 = np.array([np.power(1 / 2, 0.5), np.power(1.j / 2, 0.5)])
    x2 = (np.power(1 / 2, 0.5)) * np.array([np.sin(_theta) * np.exp(1.j * _phi), np.cos(_theta)])
    return np.power(np.abs(x1.dot(x2)), 2)



theta = np.arange(0, 2*np.pi, 0.1)
phi = np.arange(0, 2*np.pi, 0.1)
#Theta, Phi = np.meshgrid(theta, phi)
#F = f(Theta, Phi)
#G = g(Theta, Phi)

F2 = np.zeros((len(theta), len(phi)))
for i in range(63):
    for j in range(63):
        F2[j, i] = f2(theta[i], phi[j])
G2 = np.zeros((len(theta), len(phi)))
for i in range(63):
    for j in range(63):
        G2[j, i] = g2(theta[i], phi[j])

# im = plt.imshow(F, cmap=matplotlib.cm.RdBu)
# matplotlib.axes.Axes.set_xlim(right=0, left=6)
#ax1 = plt.subplot(121)
#plt.ylabel(r'$\phi$')
#plt.xlabel(r'$\theta$')
#plt.title(r'$\tau$ is zero')
#plt.axis([0, 6, 0, 6])
#plt.imshow(F, aspect='auto', extent=(Theta.min(), Theta.max(), Phi.min(), Phi.max()))

#ax1_2 = plt.subplot(122)
#plt.ylabel(r'$\phi$')
#plt.xlabel(r'$\theta$')
#plt.title(r'$\tau$ is zero')
#plt.axis([0, 6, 0, 6])
#plt.imshow(F2, aspect='auto', extent=(Theta.min(), Theta.max(), Phi.min(), Phi.max()))
#print(F2.shape)

ax1 = plt.subplot(121)
plt.ylabel(r'$\phi$')
plt.xlabel(r'$\theta$')
plt.title(r'$\tau$ is Hadamard')
plt.axis([0, 6, 0, 6])
plt.imshow(F2, aspect='auto', extent=(theta.min(), theta.max(), phi.min(), phi.max()))

ax2 = plt.subplot(122)
plt.ylabel(r'$\phi$')
plt.xlabel(r'$\theta$')
plt.title(r'$\tau$ is Hadamard')
plt.axis([0, 6, 0, 6])
plt.imshow(G2, aspect='auto', extent=(theta.min(), theta.max(), phi.min(), phi.max()))

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
