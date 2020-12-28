import numpy as np

# a) Create a vector of zero and sinusoidal components
zeros = np.zeros(500)
n = np.arange(500, 601)
sinusoid = np.cos(2 * np.pi * 0.1 * n)
y = np.concatenate([zeros, sinusoid])
y = np.concatenate([y, np.zeros(300)])

#%%
# b) Create a noisy version of the signal x[n]
y_n = y + np.sqrt(0.5) * np.random.randn(y.size)

#%%
# c) Implement the deterministic sinusoid detector
y_d = np.convolve(sinusoid, y_n, 'same')

#%%
# d)  Implement the random signal version
e = np.exp(-2 * np.pi * 1j * 0.1 * n)
y_r = np.abs( np.convolve(e, y_n, 'same') )

#%%
# e) Generate plots
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 1, figsize=(15, 7))
fig.tight_layout(pad=1.5)

ax[0].plot(y)
ax[0].title.set_text('Noiseless signal')
ax[1].plot(y_n)
ax[1].title.set_text('Noisy signal')
ax[2].plot(y_d)
ax[2].title.set_text('Deterministic detector')
ax[3].plot(y_r)
ax[3].title.set_text('Stochastic detector')


 
