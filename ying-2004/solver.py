import math
import numpy as np

import tree
import kifmm as kind
import algo


def direct(particles):
    for i in particles:
        for j in particles:
            if i != j:
                position = i.position - j.position
                r = math.sqrt(position[0]**2+position[1]**2)
                i.real_pot += j.strength/(2*math.pi)*math.log(1/r)


def error(particles):
    error_num = 0
    error_den = 0
    for particle in particles:
        error_num += (particle.real_pot - particle.pot)**2
        error_den += (particle.real_pot)**2
    
    return math.sqrt(error_num/error_den)

n=100
positions = np.random.rand(n, 2)
# positions = np.array([[0.125, 0.875], [0.125, 0.625], [0.625, 0.375], [0.625, 0.125]])

particles = []

for i in range(n):
    particles.append(tree.Particle(positions[i], 1))

t = tree.Tree()
t.build(1, np.array([0.5, 0.5]), particles, 5, 16)
algo.compute_value(t, kind)

direct(particles)

print("Error - {}".format(error(particles)))

print("Potential C - {}".format(particles[1].pot))
print("Potential A - {}".format(particles[1].real_pot))