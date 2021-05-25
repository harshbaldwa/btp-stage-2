from compyle.api import annotate, wrap, Elementwise, Reduction, Scan, get_config
from compyle.types import declare
from compyle.low_level import cast
from math import atan2, cos, log, sin, sqrt, floor
import numpy as np
import time


@annotate(double="x, y, x_min, y_min, length", b_len="int", return_="int")
def get_cell_id(x, y, b_len, x_min=0.0, y_min=0.0, length=1):
    id = declare("int")
    nx, ny = declare("int", 2)
    nx = cast(floor((b_len * 2 * (x - x_min)) / length), "int")
    ny = cast(floor((b_len * 2 * (y - y_min)) / length), "int")

    b, s = declare('matrix(4, "int")', 2)
    s[0] = 1
    s[1] = 2
    s[2] = 4
    s[3] = 8
    b[0] = 0x55555555
    b[1] = 0x33333333
    b[2] = 0x0F0F0F0F
    b[3] = 0x00FF00FF

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


@annotate(int="i, b_len", intp="bin_count, bin_offset", doublep="x, y")
def get_bin_count(i, x, y, b_len, bin_count, bin_offset):
    id = declare("int")
    id = get_cell_id(x[i], y[i], b_len, 0, 0, 1)
    bin_offset[i] = bin_count[id]
    bin_count[id] += 1


@annotate(i="int", bin_count="intp", return_="int")
def input_bin_count(i, bin_count):
    if i == 0:
        return 0
    else:
        return bin_count[i - 1]


@annotate(int="i, item", start_index="intp")
def output_bin_count(i, item, start_index):
    start_index[i] = item


@annotate(int="i, b_len", intp="bin_offset, start_index, indices", doublep="x, y")
def start_indices(i, x, y, b_len, bin_offset, start_index, indices):
    id = declare("int")
    id = get_cell_id(x[i], y[i], b_len, 0, 0, 1)
    indices[start_index[id] + bin_offset[i]] = i


@annotate(n="int", return_="int")
def _factorial(n):
    result, i = declare("int", 2)
    result = 1
    if n > 0:
        for i in range(1, n + 1):
            result *= i
    return result


@annotate(int="n, r", return_="int")
def nCr(n, r):
    result = declare("int")
    result = _factorial(n) // (_factorial(r) * _factorial(n - r))
    return result


@annotate(power="int", double="real, imag, mag_multi", new_num="doublep")
def complex_power(real, imag, power, mag_multi, new_num):
    mag, ang = declare("double", 2)
    mag = sqrt(real ** 2 + imag ** 2)
    ang = atan2(imag, real) * power
    mag = (mag ** power) * mag_multi
    new_num[0] = mag * (cos(ang))
    new_num[1] = mag * (sin(ang))


@annotate(
    power="int",
    double="real, imag, multi_real, multi_imag, multi_factor",
    new_num="doublep",
)
def complex_power_multi(
    real, imag, power, multi_real, multi_imag, multi_factor, new_num
):
    mag, ang = declare("double", 2)
    mag = (sqrt(real ** 2 + imag ** 2)) ** power
    ang = atan2(imag, real) * power
    new_num[0] = multi_factor * (
        mag * (cos(ang)) * multi_real - mag * (sin(ang)) * multi_imag
    )
    new_num[1] = multi_factor * (
        mag * (sin(ang)) * multi_real + mag * (cos(ang)) * multi_imag
    )


@annotate(double="z_real, z_imag", return_="double")
def log_complex(z_real, z_imag):
    result, abs_value = declare("double", 2)
    abs_value = sqrt(z_real ** 2 + z_imag ** 2)
    result = log(abs_value)
    return result


@annotate(
    int="i, p",
    intp="start, bin_count, indices",
    doublep="multipole_real, multipole_imag, prop_part, x_part, y_part, box_x, box_y",
)
def mulitpole_finest(
    i,
    start,
    bin_count,
    indices,
    multipole_real,
    multipole_imag,
    prop_part,
    x_part,
    y_part,
    box_x,
    box_y,
    p,
):
    box_number, j, k, pid = declare("int", 4)
    complex_num = declare("matrix(2)")
    box_number = i * (p + 1)
    for j in range(bin_count[i]):
        pid = indices[start[i] + j]
        multipole_real[box_number] += prop_part[pid]
        for k in range(1, p + 1):
            complex_power(
                x_part[pid] - box_x[i],
                y_part[pid] - box_y[i],
                k,
                -prop_part[pid] / k,
                complex_num,
            )
            multipole_real[box_number + k] += complex_num[0]
            multipole_imag[box_number + k] += complex_num[1]


@annotate(
    int="i, p",
    doublep="multipole_p_real, multipole_p_imag, multipole_c_real, multipole_c_imag, px, py, cx, cy",
)
def transfer_multipole(
    i,
    multipole_p_real,
    multipole_p_imag,
    multipole_c_real,
    multipole_c_imag,
    px,
    py,
    cx,
    cy,
    p,
):
    j, c_id, k, l, center_p_id, center_c_id = declare("int", 6)
    complex_num = declare("matrix(2)")
    mag_multi = declare("double")
    center_p_id = cast(floor(i / (p + 1)), "int")
    l = i % (p + 1)
    for j in range(4):
        center_c_id = 4 * center_p_id + j
        c_id = (p + 1) * center_c_id
        if l == 0:
            multipole_p_real[i] += multipole_c_real[c_id]
        else:
            for k in range(1, l + 1):
                mag_multi = nCr(l - 1, k - 1)
                complex_power_multi(
                    cx[center_c_id] - px[center_p_id],
                    cy[center_c_id] - py[center_p_id],
                    l - k,
                    multipole_c_real[c_id + k],
                    multipole_c_imag[c_id + k],
                    mag_multi,
                    complex_num,
                )
                multipole_p_real[i] += complex_num[0]
                multipole_p_imag[i] += complex_num[1]
            complex_power(
                cx[center_c_id] - px[center_p_id],
                cy[center_c_id] - py[center_p_id],
                l,
                multipole_c_real[c_id] / l,
                complex_num,
            )
            multipole_p_real[i] -= complex_num[0]
            multipole_p_imag[i] -= complex_num[1]


@annotate(
    int="cid, cell_level, pid, level, p",
    intp="bin_count, start, indices",
    doublep="prop_part, x_part, y_part, multipole_real, multipole_imag, cx, cy, value",
    length="double",
)
def compute_cell_value(
    cid,
    cell_level,
    pid,
    value,
    prop_part,
    x_part,
    y_part,
    multipole_real,
    multipole_imag,
    cx,
    cy,
    level,
    length,
    p,
    bin_count,
    start,
    indices,
):
    i, j = declare("int", 2)
    complex_num = declare("matrix(2)")
    if sqrt((cx[cid] - x_part[pid]) ** 2 + (cy[cid] - y_part[pid]) ** 2) > length / (
        2 ** (cell_level - 1)
    ):
        value[pid] += multipole_real[cid * (p + 1)] * log_complex(
            x_part[pid] - cx[cid], y_part[pid] - cy[cid]
        )
        for j in range(1, p + 1):
            complex_power_multi(
                x_part[pid] - cx[cid],
                y_part[pid] - cy[cid],
                -j,
                multipole_real[cid * (p + 1) + j],
                multipole_imag[cid * (p + 1) + j],
                1,
                complex_num,
            )
            value[pid] += complex_num[0]
    elif cell_level < level:
        for i in range(4):
            compute_cell_value(
                4 * (cid - 4 ** level) + i,
                cell_level + 1,
                pid,
                value,
                prop_part,
                x_part,
                y_part,
                multipole_real,
                multipole_imag,
                cx,
                cy,
                level,
                length,
                p,
                bin_count,
                start,
                indices,
            )
    else:
        for i in range(bin_count[cid]):
            if pid != indices[start[cid] + i]:
                value[pid] += prop_part[indices[start[cid] + i]] * log_complex(
                    x_part[pid] - x_part[indices[start[cid] + i]],
                    y_part[pid] - y_part[indices[start[cid] + i]],
                )


@annotate(
    int="i, level, p, total_blocks",
    intp="bin_count, start, indices",
    doublep="prop_part, x_part, y_part, multipole_real, multipole_imag, cx, cy, value",
    length="double",
)
def compute_value(
    i,
    value,
    prop_part,
    x_part,
    y_part,
    multipole_real,
    multipole_imag,
    cx,
    cy,
    level,
    length,
    p,
    bin_count,
    start,
    indices,
    total_blocks,
):
    j = declare("int")
    for j in range(16):
        compute_cell_value(
            total_blocks - 16 + j,
            2,
            i,
            value,
            prop_part,
            x_part,
            y_part,
            multipole_real,
            multipole_imag,
            cx,
            cy,
            level,
            length,
            p,
            bin_count,
            start,
            indices,
        )


if __name__ == "__main__":

    backend = "cython"
    get_config().use_openmp = True

    n = 1000
    x = np.random.random(n)
    y = np.random.random(n)
    prop_part = np.ones(n)
    x_direct = x.copy()
    y_direct = y.copy()
    prop_part_direct = prop_part.copy()
    level = 5
    length = 1
    b_len = 2 ** (level - 1)
    p = 12
    total_blocks = int(16 * (4 ** (level - 1) - 1) / 3)

    bin_count = np.zeros(4 ** level, dtype=np.int32)
    start = np.zeros_like(bin_count, dtype=np.int32)
    bin_offset = np.zeros_like(x, dtype=np.int32)
    indices = np.zeros_like(x, dtype=np.int32)

    wasteblocks = 4 ** (level + 1) * int((4 ** (12 - level) - 1) / 3)

    npzfile = np.load("centers.npz")
    cx = npzfile["cx"]
    cy = npzfile["cy"]
    cx = cx[wasteblocks:]
    cy = cy[wasteblocks:]

    multipole_real = np.zeros(total_blocks * (p + 1))
    multipole_imag = np.zeros(total_blocks * (p + 1))

    value = np.zeros_like(x)
    value_real = np.zeros_like(x)

    (
        x,
        y,
        bin_count,
        start,
        bin_offset,
        indices,
        cx,
        cy,
        multipole_real,
        multipole_imag,
        value,
    ) = wrap(
        x,
        y,
        bin_count,
        start,
        bin_offset,
        indices,
        cx,
        cy,
        multipole_real,
        multipole_imag,
        value,
        backend=backend,
    )


    eget_bin_count = Elementwise(get_bin_count, backend=backend)
    cum_bin_count = Scan(
        input_bin_count, output_bin_count, "a+b", dtype=np.int32, backend=backend
    )
    estart_indices = Elementwise(start_indices, backend=backend)
    emultipole_finest = Elementwise(mulitpole_finest, backend=backend)
    etransfer_multipole = Elementwise(transfer_multipole, backend=backend)
    ecompute_value = Elementwise(compute_value, backend=backend)


    time_start_fmm = time.time()

    eget_bin_count(x, y, b_len, bin_count, bin_offset)
    cum_bin_count(bin_count=bin_count, start_index=start)
    estart_indices(x, y, b_len, bin_offset, start, indices)
    emultipole_finest(
        start,
        bin_count,
        indices,
        multipole_real,
        multipole_imag,
        prop_part,
        x,
        y,
        cx,
        cy,
        p,
    )
    for l in range(level, 2, -1):
        index1 = int((4 ** (level - l) - 1) / 3) * 4 ** (l + 1)
        index2 = int((4 ** (level - l + 1) - 1) / 3) * 4 ** l
        index3 = int((4 ** (level - l + 2) - 1) / 3) * 4 ** (l - 1)
        etransfer_multipole(
            multipole_real[index2 * (p + 1) : index3 * (p + 1)],
            multipole_imag[index2 * (p + 1) : index3 * (p + 1)],
            multipole_real[index1 * (p + 1) : index2 * (p + 1)],
            multipole_imag[index1 * (p + 1) : index2 * (p + 1)],
            cx[index2:index3],
            cy[index2:index3],
            cx[index1:index2],
            cy[index1:index2],
            p,
        )

    ecompute_value(
        value,
        prop_part,
        x,
        y,
        multipole_real,
        multipole_imag,
        cx,
        cy,
        level,
        length,
        p,
        bin_count,
        start,
        indices,
        total_blocks,
    )

    time_stop_fmm = time.time()


    print("done!")


    time_start_direct = time.time()

    for i in range(n):
        for j in range(n):
            if i != j:
                value_real[i] += prop_part_direct[j] * np.log(
                    abs((x_direct[i] - x_direct[j]) + 1j * (y_direct[i] - y_direct[j]))
                )


    time_stop_direct = time.time()

    error = np.sum((value_real - value) ** 2) / np.sum(value_real ** 2)
    print("Error - ", error)
    print("Time (by fmm) - ", time_stop_fmm - time_start_fmm)
    print("Time (by direct) - ", time_stop_direct - time_start_direct)
    print(
        "Speedup - ",
        (time_stop_direct - time_start_direct) / (time_stop_fmm - time_start_fmm),
    )
