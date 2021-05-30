import numpy as np
from compyle.api import annotate, wrap, Elementwise, Reduction, Scan, get_config
from .main import get_bin_count, input_bin_count, output_bin_count, start_indices, multipole_finest, transfer_multipole, compute_value

backend = 'cython'

def initialise_tree():
    x = np.array([0.2, 0.25, 0.25, 0.75, 0.75])
    y = np.array([0.2, 0.25, 0.75, 0.25, 0.75])
    return x, y

def initialise_fmm():
    x = np.array([0.12, 0.88])
    y = np.array([0.12, 0.88])
    prop = np.array([1.0, 1.0])
    return x, y, prop

def test_bin_count():
    bin_count = np.zeros(4, dtype=np.int32)
    exp_bin_count = np.array([2, 1, 1, 1], dtype=np.int32)
    bin_offset = np.zeros(5, dtype=np.int32)
    exp_bin_offset = np.array([0, 1, 0, 0, 0], dtype=np.int32)
    b_len = 1
    x, y = initialise_tree()
    eget_bin_count = Elementwise(get_bin_count, backend=backend)
    eget_bin_count(x, y, b_len, bin_count, bin_offset)

    np.testing.assert_array_equal(bin_count, exp_bin_count)
    np.testing.assert_array_equal(bin_offset, exp_bin_offset)

def test_cum_bin_count():
    start = np.zeros(4, dtype=np.int32)
    exp_start = np.array([0, 2, 3, 4], dtype=np.int32)
    bin_count = np.array([2, 1, 1, 1], dtype=np.int32)
    x, y = initialise_tree()
    cum_bin_count = Scan(
        input_bin_count, output_bin_count, "a+b", dtype=np.int32, backend=backend
    )
    cum_bin_count(bin_count=bin_count, start_index=start)

    np.testing.assert_array_equal(start, exp_start)

def test_start_index():
    start = np.array([0, 2, 3, 4], dtype=np.int32)
    bin_count = np.array([2, 1, 1, 1], dtype=np.int32)
    bin_offset = np.array([0, 1, 0, 0, 0], dtype=np.int32)
    indices = np.zeros(5, dtype=np.int32)
    b_len = 1
    exp_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    x, y = initialise_tree()
    estart_indices = Elementwise(start_indices, backend=backend)
    estart_indices(x, y, b_len, bin_offset, start, indices)

    np.testing.assert_array_equal(indices, exp_indices)

def test_multipole_finest():
    level = 2
    p = 1
    b_len = 2**(level-1)
    bin_count = np.zeros(4**level ,dtype=np.int32)
    start = np.zeros(4**level ,dtype=np.int32)
    bin_offset = np.zeros(2, dtype=np.int32)
    indices = np.zeros(2, dtype=np.int32)
    total_blocks = int(16 * (4 ** (level - 1) - 1) / 3)
    wasteblocks = 4 ** (level + 1) * int((4 ** (12 - level) - 1) / 3)
    npzfile = np.load("compyle_implementation/centers.npz")
    cx = npzfile["cx"]
    cy = npzfile["cy"]
    cx = cx[wasteblocks:]
    cy = cy[wasteblocks:]
    multipole_real = np.zeros(total_blocks * (p + 1))
    multipole_imag = np.zeros(total_blocks * (p + 1))
    exp_multipole_real = np.zeros(total_blocks * (p + 1))
    exp_multipole_imag = np.zeros(total_blocks * (p + 1))
    exp_multipole_real[0] = 1
    exp_multipole_real[1] = 0.005
    exp_multipole_imag[0] = 0
    exp_multipole_imag[1] = 0.005
    exp_multipole_real[15*(p+1)+0] = 1
    exp_multipole_real[15*(p+1)+1] = -0.005
    exp_multipole_imag[15*(p+1)+0] = 0
    exp_multipole_imag[15*(p+1)+1] = -0.005
    x, y, prop = initialise_fmm()
    eget_bin_count = Elementwise(get_bin_count, backend=backend)
    cum_bin_count = Scan(
        input_bin_count, output_bin_count, "a+b", dtype=np.int32, backend=backend
    )
    estart_indices = Elementwise(start_indices, backend=backend)

    eget_bin_count(x, y, b_len, bin_count, bin_offset)
    cum_bin_count(bin_count=bin_count, start_index=start)
    estart_indices(x, y, b_len, bin_offset, start, indices)
    emultipole_finest = Elementwise(multipole_finest, backend=backend)
    emultipole_finest(multipole_real[:4**level*(p+1)], multipole_imag[:4**level*(p+1)], start, bin_count, indices, x, y, prop, cx[:4**level], cy[:4**level], p)

    np.testing.assert_array_almost_equal(multipole_real, exp_multipole_real)
    np.testing.assert_array_almost_equal(multipole_imag, exp_multipole_imag)

def test_transfer_multipole():
    level = 2
    p = 1
    b_len = 2**(level-1)
    bin_count = np.zeros(4**level ,dtype=np.int32)
    start = np.zeros(4**level ,dtype=np.int32)
    bin_offset = np.zeros(2, dtype=np.int32)
    indices = np.zeros(2, dtype=np.int32)
    total_blocks = int(4 * (4 ** (level) - 1) / 3)
    wasteblocks = 4 ** (level + 1) * int((4 ** (12 - level) - 1) / 3)
    npzfile = np.load("compyle_implementation/centers.npz")
    cx = npzfile["cx"]
    cy = npzfile["cy"]
    cx = cx[wasteblocks:]
    cy = cy[wasteblocks:]
    cx = np.append(cx, [0.25, 0.25, 0.75, 0.75])
    cy = np.append(cy, [0.25, 0.75, 0.25, 0.75])
    multipole_real = np.zeros(total_blocks * (p + 1))
    multipole_imag = np.zeros(total_blocks * (p + 1))
    exp_multipole_real = np.zeros(total_blocks * (p + 1))
    exp_multipole_imag = np.zeros(total_blocks * (p + 1))
    
    exp_multipole_real[0] = 1
    exp_multipole_real[1] = 0.005
    exp_multipole_imag[0] = 0
    exp_multipole_imag[1] = 0.005
    exp_multipole_real[15*(p+1)+0] = 1
    exp_multipole_real[15*(p+1)+1] = -0.005
    exp_multipole_imag[15*(p+1)+0] = 0
    exp_multipole_imag[15*(p+1)+1] = -0.005
    
    exp_multipole_real[16*(p+1)+0] = 1
    exp_multipole_imag[16*(p+1)+0] = 0
    exp_multipole_real[16*(p+1)+1] = 0.13
    exp_multipole_imag[16*(p+1)+1] = 0.13
    exp_multipole_real[19*(p+1)+0] = 1
    exp_multipole_imag[19*(p+1)+0] = 0
    exp_multipole_real[19*(p+1)+1] = -0.13
    exp_multipole_imag[19*(p+1)+1] = -0.13

    x, y, prop = initialise_fmm()
    eget_bin_count = Elementwise(get_bin_count, backend=backend)
    cum_bin_count = Scan(
        input_bin_count, output_bin_count, "a+b", dtype=np.int32, backend=backend
    )
    estart_indices = Elementwise(start_indices, backend=backend)
    emultipole_finest = Elementwise(multipole_finest, backend=backend)
    etransfer_multipole = Elementwise(transfer_multipole, backend=backend)

    eget_bin_count(x, y, b_len, bin_count, bin_offset)
    cum_bin_count(bin_count=bin_count, start_index=start)
    estart_indices(x, y, b_len, bin_offset, start, indices)
    emultipole_finest(multipole_real[:4**level*(p+1)], multipole_imag[:4**level*(p+1)], start, bin_count, indices, x, y, prop, cx[:4**level], cy[:4**level], p)
    etransfer_multipole(multipole_real[16*(p+1):], multipole_imag[16*(p+1):], multipole_real[:16*(p+1)], multipole_imag[:16*(p+1)], cx[16:], cy[16:], cx[:16], cy[:16], p)

    np.testing.assert_array_almost_equal(multipole_real, exp_multipole_real)
    np.testing.assert_array_almost_equal(multipole_imag, exp_multipole_imag)
    
def test_compute_value():
    level = 2
    p = 1
    b_len = 2**(level-1)
    length = 1
    bin_count = np.zeros(4**level ,dtype=np.int32)
    start = np.zeros(4**level ,dtype=np.int32)
    bin_offset = np.zeros(2, dtype=np.int32)
    indices = np.zeros(2, dtype=np.int32)
    total_blocks = int(16 * (4 ** (level - 1) - 1) / 3)
    wasteblocks = 4 ** (level + 1) * int((4 ** (12 - level) - 1) / 3)
    npzfile = np.load("compyle_implementation/centers.npz")
    cx = npzfile["cx"]
    cy = npzfile["cy"]
    cx = cx[wasteblocks:]
    cy = cy[wasteblocks:]
    multipole_real = np.zeros(total_blocks * (p + 1))
    multipole_imag = np.zeros(total_blocks * (p + 1))
    value = np.zeros(2)
    exp_value = np.zeros(2)

    z0 = 0.12 + 0.12*1j - (0.875 + 0.875*1j)
    z1 = 0.88 + 0.88*1j - (0.125 + 0.125*1j)
    exp_value[0] = (1*np.log(z0) - 0.005*(1+1j)*(z0**-1)).real
    exp_value[1] = (1*np.log(z1) + 0.005*(1+1j)*(z1**-1)).real
    x, y, prop = initialise_fmm()
    eget_bin_count = Elementwise(get_bin_count, backend=backend)
    cum_bin_count = Scan(
        input_bin_count, output_bin_count, "a+b", dtype=np.int32, backend=backend
    )
    estart_indices = Elementwise(start_indices, backend=backend)
    emultipole_finest = Elementwise(multipole_finest, backend=backend)
    ecompute_value = Elementwise(compute_value, backend=backend)

    eget_bin_count(x, y, b_len, bin_count, bin_offset)
    cum_bin_count(bin_count=bin_count, start_index=start)
    estart_indices(x, y, b_len, bin_offset, start, indices)
    emultipole_finest(multipole_real[:4**level*(p+1)], multipole_imag[:4**level*(p+1)], start, bin_count, indices, x, y, prop, cx[:4**level], cy[:4**level], p)
    ecompute_value(value,prop,x,y,multipole_real,multipole_imag,cx,cy,level,length,p,bin_count,start,indices,total_blocks)

    np.testing.assert_array_almost_equal(value, exp_value)
