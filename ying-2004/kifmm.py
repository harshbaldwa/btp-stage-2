import math
import numpy as np

from . import tree


def direct(particles, point):
    value = 0
    for particle in particles:
        r = point - particle.position
        dist = (r[0]**2 + r[1]**2)**0.5

        value += particle.strength/(2*math.pi)*math.log(1/dist)

    return value

def S2M(cell):
    n = cell.surface_number
    check_pot = np.zeros(n)
    for i in range(n):
        check_pot[i] += direct(cell.particles, cell.outer_circle[i])
    
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = cell.outer_circle[i] - cell.inner_circle[j]
            dist = (r[0]**2 + r[1]**2)**0.5
            matrix[i][j] = 1/(2*math.pi)*math.log(1/dist)
    
    inner_particles = np.linalg.solve(matrix, check_pot)

    for i in range(n):
        cell.inner_equi[i].strength = inner_particles[i]

def M2M_check(cell):
    n = cell.surface_number
    for i in range(n):
        cell.parent.outer_check[i].pot += direct(cell.inner_equi, cell.parent.outer_circle[i])

def M2M_equi(cell):
    n = cell.surface_number
    check_pot = np.zeros(n)
    matrix = np.zeros((n, n))
    for i in range(n):
        check_pot[i] = cell.outer_check[i].pot
        for j in range(n):
            r = cell.outer_circle[i] - cell.inner_circle[j]
            dist = (r[0]**2 + r[1]**2)**0.5
            matrix[i][j] = 1/(2*math.pi)*math.log(1/dist)

    outer_particles = np.linalg.solve(matrix, check_pot)

    for i in range(n):
        cell.inner_equi[i].strength = outer_particles[i]

def M2L(cell, other):
    n = cell.surface_number
    for i in range(n):
        other.inner_check[i].pot = direct(cell.inner_equi, other.inner_circle[i])

    check_pot = np.zeros(n)
    matrix = np.zeros((n, n))
    for i in range(n):
        check_pot[i] = other.inner_check[i].pot
        for j in range(n):
            r = other.inner_circle[i] - other.outer_circle[j]
            dist = (r[0]**2 + r[1]**2)**0.5
            matrix[i][j] = 1/(2*math.pi)*math.log(1/dist)
    
    outer_particles = np.linalg.solve(matrix, check_pot)

    for i in range(n):
        cell.outer_equi[i].strength = outer_particles[i]

def L2L(cell):
    n = cell.surface_number
    for child in cell.children:
        check_pot = np.zeros(n)
        for i in range(n):
            check_pot[i] += direct(cell.outer_equi, child.inner_circle[i])

        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                r = child.inner_circle[i] - child.outer_circle[j]
                dist = (r[0]**2 + r[1]**2)**0.5
                matrix[i][j] = 1/(2*math.pi)*math.log(1/dist)

        outer_particles = np.linalg.solve(matrix, check_pot)

        for i in range(n):
            cell.outer_equi[i].strength = outer_particles[i]

