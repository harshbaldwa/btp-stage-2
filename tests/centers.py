import numpy as np

LEVEL = 12

total_blocks = 0
for level in range(LEVEL, 1, -1):
    total_blocks += 4**level

cx = np.zeros(total_blocks)
cy = np.zeros(total_blocks)

offset = 0

for level in range(LEVEL, 1, -1):
    for idx in range(4**level):
        a = format(idx, '032b')
        x = int(a[::2], 2)
        y = int(a[1::2], 2)
        cx[offset+idx] = (x+0.5)/(2**level)
        cy[offset+idx] = (y+0.5)/(2**level)
    offset += 4**level

np.savez_compressed("centers", cx=cx, cy=cy)
