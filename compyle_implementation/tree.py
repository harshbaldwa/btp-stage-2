from compyle.api import annotate, wrap, Elementwise, Reduction, Scan
from compyle.types import declare
from math import floor
import numpy as np

backend = 'cython'

@annotate(b_len='int', intp='b1, b2, inter')
def bits_interleaving(b_len, b1, b2, inter):
    i = declare('int')
    for i in range(2*b_len):
        if i%2 == 0:
            inter[i] = b1[i//2]
        else:
            inter[i] = b2[i//2]

@annotate(int='n, b_len', return_='intp')
def dec2bin(n, b_len):
    binary = declare('matrix(1, "int")')
    # binary = declare('matrix({}, "int")'.format(b_len))
    i = declare('int')
    for i in range(b_len):
        binary[b_len-i-1] = n%2
        n = n//2

    return binary


@annotate(b_len='int', b='intp', return_='int')
def bin2dec(b, b_len):
    n, i = declare('int', 2)
    n = 0
    for i in range(b_len):
        n += b[i]*2**(b_len-i-1)
    
    return n

@annotate(double='x, y, x_min, y_min, length', b_len='int', return_='int')
def get_cell_id(x, y, b_len, x_min=0.0, y_min=0.0, length=1):
    id = declare('int')
    bx, by = declare('matrix(1, "int")', 2)
    inter = declare('matrix(2, "int")')
    # how to have variable size of array declared inside
    # bx, by = declare('matrix({})'.format(b_len), 2)
    # inter = declare('matrix({})'.format(2*b_len))
    nx, ny = declare('int', 2)
    # how to make this int?
    nx = (b_len*2*(x-x_min)) // length
    ny = (b_len*2*(y-y_min)) // length
    bx = dec2bin(nx, b_len)
    by = dec2bin(ny, b_len)
    bits_interleaving(b_len, bx, by, inter)
    id = bin2dec(inter, 2*b_len)
    return id


@annotate(int='i, b_len', intp='bin_count, bin_offset', doublep='x, y')
def get_bin_count(i, x, y, b_len, bin_count, bin_offset):
    id = declare('int')
    id = get_cell_id(x[i], y[i], b_len, 0, 0, 1)
    bin_offset[i] = bin_count[id]
    bin_count[id] += 1

x = np.array([0.25, 0.75, 0.75, 0.25, 0.2])
y = np.array([0.75, 0.25, 0.75, 0.25, 0.2])
level = 1
b_len = 2**(level-1)

bin_count = np.zeros(4**level, dtype=np.int32)
start = np.zeros_like(bin_count, dtype=np.int32)
bin_offset = np.zeros_like(x, dtype=np.int32)
indices = np.zeros_like(x, dtype=np.int32)

x, y, bin_count, start, bin_offset, indices = wrap(x, y, bin_count, start, bin_offset, indices, backend=backend)
eget_bin_count = Elementwise(get_bin_count, backend=backend)
eget_bin_count(x, y, b_len, bin_count, bin_offset)

print(bin_count)