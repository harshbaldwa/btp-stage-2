import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0, 1, 17)

for i in range(17):
    plt.plot([a[i], a[i]], [0, 1], 'b')

for i in range(17):
    plt.plot([0, 1], [a[i], a[i]], 'b')

plt.plot([0.03125, 0.96875, 0.90625], [0.03125, 0.90625, 0.96875], 'r.')

plt.plot([0.875], [0.875], 'g.')

plt.show()