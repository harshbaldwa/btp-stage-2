from compyle.api import annotate, wrap, Elementwise, Reduction, Scan
from compyle.types import declare
from compyle.low_level import cast
from math import floor
import numpy as np

backend = 'cython'


@annotate(double='x, y, x_min, y_min, length', b_len='int', return_='int')
def get_cell_id(x, y, b_len, x_min=0.0, y_min=0.0, length=1):
    id = declare('int')
    nx, ny = declare('int', 2)
    nx = cast(floor((b_len*2*(x-x_min)) / length), 'int')
    ny = cast(floor((b_len*2*(y-y_min)) / length), 'int')
    
    b, s = declare('matrix(4, "int")', 2)
    s = [1, 2, 4, 8]
    b = [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF]
        
    nx = (nx | (nx << s[3])) & b[3]
    nx = (nx | (nx << s[2])) & b[2]
    nx = (nx | (nx << s[1])) & b[1]
    nx = (nx | (nx << s[0])) & b[0]
    
    ny = (ny | (ny << s[3])) & b[3]
    ny = (ny | (ny << s[2])) & b[2]
    ny = (ny | (ny << s[1])) & b[1]
    ny = (ny | (ny << s[0])) & b[0]

    id = (nx << 1) | ny
    return id


@annotate(int='i, b_len', intp='bin_count, bin_offset', doublep='x, y')
def get_bin_count(i, x, y, b_len, bin_count, bin_offset):
    id = declare('int')
    id = get_cell_id(x[i], y[i], b_len, 0, 0, 1)
    bin_offset[i] = bin_count[id]
    bin_count[id] += 1


@annotate(i='int', bin_count='intp', return_='int')
def input_bin_count(i, bin_count):
    if i==0:
        return 0
    else:
        return bin_count[i-1]


@annotate(int='i, item', start='intp')
def output_bin_count(i, item, start):
    start[i] = item


@annotate(int='i, b_len', intp='bin_offset, start, indices', doublep='x, y')
def start_indices(i, x, y, b_len, bin_offset, start, indices):
    id = declare('int')
    id = get_cell_id(x[i], y[i], b_len, 0, 0, 1)
    indices[start[id] + bin_offset[i]] = i



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


cum_bin_count = Scan(input_bin_count, output_bin_count, 'a+b', dtype=np.int32, backend=backend)
cum_bin_count(bin_count=bin_count, start=start)

estart_indices = Elementwise(start_indices, backend=backend)
estart_indices(x, y, b_len, bin_offset, start, indices)

print(bin_count)
print(start)
print(bin_offset)
print(indices)