import numpy as np

LEVEL = 8

total_blocks = 0
for level in range(LEVEL, 0, -1):
    total_blocks += 8**level

cx = np.zeros(total_blocks)
cy = np.zeros(total_blocks)
cz = np.zeros(total_blocks)
level_matrix = np.zeros(total_blocks)

offset = 0

for level in range(LEVEL, 0, -1):
    for idx in range(8**level):
        a = format(idx, '060b')
        x = int(a[2::3], 2)
        y = int(a[1::3], 2)
        z = int(a[::3], 2)
        cx[offset+idx] = (x+0.5)/(2**level)
        cy[offset+idx] = (y+0.5)/(2**level)
        cz[offset+idx] = (z+0.5)/(2**level)
        level_matrix[offset+idx] = level

    offset += 8**level

np.savez_compressed("centers", cx=cx, cy=cy, cz=cz, level_matrix=level_matrix)
