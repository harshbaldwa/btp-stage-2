# import math

# def dilate(value):
#     x = value & 0xFFFFFFFFFFFFFFF
#     x = ((x << 32) + x) & 0xFFFF00000000FFFF
#     x = ((x << 16) + x) & 0x00FF0000FF0000FF
#     x = ((x << 8) + x) & 0xF00F00F00F00F00F
#     x = ((x << 4) + x) & 0x30C30C30C30C30C3
#     x = ((x << 2) + x) & 0x9249249249249249
    
#     return x

# x = 0.99
# y = 0.99
# z = 0.99
# level = 3
# x_min = 0
# y_min = 0
# z_min = 0
# length = 1
# b_len = 2 ** level

# nx = math.floor((b_len * (x - x_min)) / length)
# ny = math.floor((b_len * (y - y_min)) / length)
# nz = math.floor((b_len * (z - z_min)) / length)

# nx = dilate(nx)
# ny = dilate(ny)
# nz = dilate(nz)

# print(bin(nx))
# print(bin(ny))
# print(bin(nz))

# value = (nz << 2) | (ny << 1) | nx

# print(bin(value))

# value0 = value & 0x3FFFFFFF
# value1 = (value >> 30)

# print(value0)
# print(value1)

import numpy as np
import math

def get_cell_id(x, y, z, b_len, x_min=0.0, y_min=0.0, z_min=0.0, length=1):
    nx = math.floor((b_len * (x - x_min)) / length)
    ny = math.floor((b_len * (y - y_min)) / length)
    nz = math.floor((b_len * (z - z_min)) / length)

    nx = nx & 0xFFFFFFFFFFFFFFF
    nx = ((nx << 32) + nx) & 0xFFFF00000000FFFF
    nx = ((nx << 16) + nx) & 0x00FF0000FF0000FF
    nx = ((nx << 8) + nx) & 0xF00F00F00F00F00F
    nx = ((nx << 4) + nx) & 0x30C30C30C30C30C3
    nx = ((nx << 2) + nx) & 0x9249249249249249

    ny = ny & 0xFFFFFFFFFFFFFFF
    ny = ((ny << 32) + ny) & 0xFFFF00000000FFFF
    ny = ((ny << 16) + ny) & 0x00FF0000FF0000FF
    ny = ((ny << 8) + ny) & 0xF00F00F00F00F00F
    ny = ((ny << 4) + ny) & 0x30C30C30C30C30C3
    ny = ((ny << 2) + ny) & 0x9249249249249249

    nz = nz & 0xFFFFFFFFFFFFFFF
    nz = ((nz << 32) + nz) & 0xFFFF00000000FFFF
    nz = ((nz << 16) + nz) & 0x00FF0000FF0000FF
    nz = ((nz << 8) + nz) & 0xF00F00F00F00F00F
    nz = ((nz << 4) + nz) & 0x30C30C30C30C30C3
    nz = ((nz << 2) + nz) & 0x9249249249249249

    id = (nz << 2) | (ny << 1) | nx
    return id



npzfile = np.load("centers.npz")
cx = npzfile["cx"]
cy = npzfile["cy"]
cz = npzfile["cz"]


