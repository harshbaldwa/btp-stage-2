import math
import numpy as np


class Particle():

    def __init__(self, position, strength):
        self.position = position
        self.strength = strength
        self.pot = 0
        self.real_pot = 0

    def reset(self):
        self.pot = 0
        self.real_pot = 0


class Cell():

    def __init__(self, level, size, center, particles, cell_limit, surface_number, surface, parent=None):
        self.level = level
        self.size = size
        self.center = center
        self.particles = particles
        self.cell_limit = cell_limit
        self.surface_number = surface_number
        self.surface = surface
        self.inner_circle = center + (math.sqrt(2))*(size/2)*surface
        self.outer_circle = center + (4 - math.sqrt(2))*(size/2)*surface
        self.inner_check = [Particle(point, 0) for point in self.outer_circle]
        self.outer_check = [Particle(point, 0) for point in self.inner_circle]
        self.inner_equi = [Particle(point, 0) for point in self.outer_circle]
        self.outer_equi = [Particle(point, 0) for point in self.inner_circle]
        self.parent = parent
        self.children = []
        self.signs = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        self.near_cells = []
        # self.children = [None]*8
        # self.signs = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],[1, 1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])

    def split(self):
        if len(self.particles) > self.cell_limit:
            for i in range(4):
            # for i in range(8):
                self.children.append(Cell(self.level+1, self.size/2, self.center+self.size/4*self.signs[i], [particle for particle in self.particles if all((particle.position-self.center)*self.signs[i] >= 0)], self.cell_limit, self.surface_number, self.surface, self))
            
            for i in range(4):
            # for i in range(8):
                self.children[i].near_cells.extend(self.children[:i]+self.children[i+1:])

            for child in self.children:
                child.split()

    def is_parent(self):
        return not len(self.children) == 0
    
    def depth(self):
        if not self.is_parent():
            return 0
        else:
            d1 = self.children[0].depth()
            d2 = self.children[1].depth()
            d3 = self.children[2].depth()
            d4 = self.children[3].depth()
            # d5 = self.children[4].depth()
            # d6 = self.children[5].depth()
            # d7 = self.children[6].depth()
            # d8 = self.children[7].depth()
            

            return max(d1, d2, d3, d4) + 1
            # return max(d1, d2, d3, d4, d5, d6, d7) + 1
    
    def childless(self, cells):
        if all([not self.is_parent(), self.level >= 1, len(self.particles) > 0]):
            cells.append(self)
        else:
            for child in self.children:
                child.childless(cells)

    def levels(self, cells):
        if len(cells) <= self.level:
            cells.append([self])
        else:
            cells[self.level].append(self)

        for child in self.children:
            child.levels(cells)
    
    def near_range(self, cell):
        dist = self.center - cell.center
        cond1 = abs(dist[0]) <= 3*self.size/2
        cond2 = abs(dist[1]) <= 3*self.size/2

        return cond1 and cond2

    def near(self):
        if self.parent:
            for near_cell in self.parent.near_cells:
                if not near_cell.is_parent():
                    if self.near_range(near_cell):
                        self.near_cells.append(near_cell)
                else:
                    for child in near_cell.children:
                        if self.near_range(child):
                            self.near_cells.append(child)

        if self.is_parent():
            for child in self.children:
                child.near()


class Tree():

    def __init__(self):
        self.cells = []
        self.childless = []
        self.depth = 0

    def build(self, size, center, particles, cell_limit, surface_number):
        theta = np.linspace(0, 2*np.pi, surface_number, False)
        surface = np.array([np.cos(theta), np.sin(theta)]).T
        root = Cell(0, size, center, particles, cell_limit, surface_number, surface)

        root.split()
        self.depth = root.depth()
        root.childless(self.childless)
        root.levels(self.cells)
        root.near()