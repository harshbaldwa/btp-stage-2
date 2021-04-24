import numpy as np

n = 6

particles = np.array((n, 3))

particles[:, 0] = 1
particles[:, 1:] = np.random.random((n, 2))

