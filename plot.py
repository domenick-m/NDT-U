import numpy as np
import matplotlib.pyplot as plt

test = np.load('test.npy')

print(test.shape[0] / 25)

fig, axes = plt.subplots()
neuron = 8

plt.plot(test[:81, neuron])


plt.tight_layout()
plt.savefig(f'test.png',facecolor='white', transparent=False)