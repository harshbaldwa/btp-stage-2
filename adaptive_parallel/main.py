from compyle.api import annotate, wrap, Elementwise, Scan, get_config
from compyle.types import declare
from compyle.low_level import cast, atomic_inc
from compyle.parallel import serial
from math import sqrt, floor

from spherical_points import spherical_points
from scipy.special import legendre
import numpy as np
import time
import argparse

# @annotate(double="x, y, z, x_min, y_min, z_min, length", b_len="int", return_="long")
# def get_cell_id(x, y, z, b_len, x_min=0.0, y_min=0.0, z_min=0.0, length=1):
#     nx, ny, nz, id = declare("int", 4)
#     nx = cast(floor((b_len * (x - x_min)) / length), "int")
#     ny = cast(floor((b_len * (y - y_min)) / length), "int")
#     nz = cast(floor((b_len * (z - z_min)) / length), "int")

#     nx = nx & 0xFFFFFFFFFFFFFFF
#     nx = ((nx << 32) + nx) & 0xFFFF00000000FFFF
#     nx = ((nx << 16) + nx) & 0x00FF0000FF0000FF
#     nx = ((nx << 8) + nx) & 0xF00F00F00F00F00F
#     nx = ((nx << 4) + nx) & 0x30C30C30C30C30C3
#     nx = ((nx << 2) + nx) & 0x9249249249249249

#     ny = ny & 0xFFFFFFFFFFFFFFF
#     ny = ((ny << 32) + ny) & 0xFFFF00000000FFFF
#     ny = ((ny << 16) + ny) & 0x00FF0000FF0000FF
#     ny = ((ny << 8) + ny) & 0xF00F00F00F00F00F
#     ny = ((ny << 4) + ny) & 0x30C30C30C30C30C3
#     ny = ((ny << 2) + ny) & 0x9249249249249249

#     nz = nz & 0xFFFFFFFFFFFFFFF
#     nz = ((nz << 32) + nz) & 0xFFFF00000000FFFF
#     nz = ((nz << 16) + nz) & 0x00FF0000FF0000FF
#     nz = ((nz << 8) + nz) & 0xF00F00F00F00F00F
#     nz = ((nz << 4) + nz) & 0x30C30C30C30C30C3
#     nz = ((nz << 2) + nz) & 0x9249249249249249

#     id = (nz << 2) | (ny << 1) | nx
#     return id

@annotate(double="x, y, z, x_min, y_min, z_min, length", b_len="int", return_="int")
def get_cell_id(x, y, z, b_len, x_min=0.0, y_min=0.0, z_min=0.0, length=1):
    nx, ny, nz, id = declare("int", 4)
    nx = cast(floor((b_len * (x - x_min)) / length), "int")
    ny = cast(floor((b_len * (y - y_min)) / length), "int")
    nz = cast(floor((b_len * (z - z_min)) / length), "int")

    nx = (nx | (nx << 16)) & 0x030000FF
    nx = (nx | (nx <<  8)) & 0x0300F00F
    nx = (nx | (nx <<  4)) & 0x030C30C3
    nx = (nx | (nx <<  2)) & 0x09249249

    ny = (ny | (ny << 16)) & 0x030000FF
    ny = (ny | (ny <<  8)) & 0x0300F00F
    ny = (ny | (ny <<  4)) & 0x030C30C3
    ny = (ny | (ny <<  2)) & 0x09249249

    nz = (nz | (nz << 16)) & 0x030000FF
    nz = (nz | (nz <<  8)) & 0x0300F00F
    nz = (nz | (nz <<  4)) & 0x030C30C3
    nz = (nz | (nz <<  2)) & 0x09249249

    id = (nz << 2) | (ny << 1) | nx
    return id


@annotate(i='int', bin_count='intp')
def initial_bin_count(i, bin_count):
    bin_count[i] = 0


@serial
@annotate(int="i, b_len", intp="bin_count, bin_offset", doublep="x, y, z")
def get_bin_count(i, x, y, z, b_len, bin_count, bin_offset):
    id = declare("int")
    idx = declare("int")
    id = get_cell_id(x[i], y[i], z[i], b_len, 0, 0, 0, 1)
    idx = atomic_inc(bin_count[id])
    bin_offset[i] = idx


@annotate(i="int", bin_count="intp", return_="int")
def input_bin_count(i, bin_count):
    if i == 0:
        return 0
    else:
        return bin_count[i - 1]


@annotate(int="i, item", start_index="intp")
def output_bin_count(i, item, start_index):
    start_index[i] = item


@annotate(
    int="i, b_len",
    intp="bin_offset, indices, start_index",
    doublep="x, y, z"
)
def start_indices(i, x, y, z, b_len, bin_offset, start_index, indices):
    id = declare("int")
    id = get_cell_id(x[i], y[i], z[i], b_len, 0, 0, 0, 1)
    indices[start_index[id] + bin_offset[i]] = i


@annotate(
    int="i, number_makino",
    doublep="cx, cy, cz, outer_x, outer_y, outer_z, inner_x, inner_y, inner_z, sph_points",
    level="intp",
    length="double"
)
def setting_pseudoparticles(
    i, cx, cy, cz, outer_x, outer_y, outer_z, inner_x, inner_y, inner_z, 
    sph_points, length, level, number_makino
):
    j = declare("int")
    size_cell = declare("double")
    size_cell = length/(2**(level[i]+1))
    for j in range(number_makino):
        outer_x[i*number_makino + j] = cx[i] + sph_points[3*j]*3*size_cell
        outer_y[i*number_makino + j] = cy[i] + sph_points[3*j+1]*3*size_cell
        outer_z[i*number_makino + j] = cz[i] + sph_points[3*j+2]*3*size_cell
        inner_x[i*number_makino + j] = cx[i] + sph_points[3*j]*0.5*size_cell
        inner_y[i*number_makino + j] = cy[i] + sph_points[3*j+1]*0.5*size_cell
        inner_z[i*number_makino + j] = cz[i] + sph_points[3*j+2]*0.5*size_cell


@annotate(int="l_len, start_id", x="double", coeff_list="doublep", return_="double")
def calc_legendre(coeff_list, x, l_len, start_id):
    i = declare("int")
    result = declare("double")
    result = 0
    for i in range(start_id, start_id+l_len):
        result += coeff_list[i]*(x**i)

    return result


# remove unnecessary calculations, position of makino pseudoparticles!!
# done
@annotate(
    int="i, level, l_limit, number_makino",
    intp="bin_count, indices, start",
    doublep="pseudo_value, pseudo_x, pseudo_y, pseudo_z, part_value, part_x, part_y, part_z, cx, cy, cz, l_list",
    length="double"
)
def calc_pseudoparticles_fine(
    i, pseudo_value, pseudo_x, pseudo_y, pseudo_z, part_value, 
    part_x, part_y, part_z, cx, cy, cz, indices, start,
    bin_count, number_makino, length, level, l_limit, l_list
):
    j, l, cid, pid, start_id = declare("int", 5)
    p2c, m2c = declare("matrix(3)", 2)
    m2c_l, p2c_l, cos_gamma, rr, l_result, pseudo_result = declare("double", 6)
    cid = cast(floor(i / number_makino), "int")
    m2c[0] = pseudo_x[i] - cx[cid]
    m2c[1] = pseudo_y[i] - cy[cid]
    m2c[2] = pseudo_z[i] - cz[cid]
    m2c_l = length/(2**(level+1))*3
    pseudo_result = 0
    for j in range(bin_count[cid]):
        pid = indices[start[cid] + j]
        p2c[0] = part_x[pid] - cx[cid]
        p2c[1] = part_y[pid] - cy[cid]
        p2c[2] = part_z[pid] - cz[cid]
        p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
        pseudo_result += part_value[pid]
        if p2c_l != 0:
            rr = p2c_l / m2c_l
            cos_gamma = (m2c[0]*p2c[0] + m2c[1]*p2c[1] + m2c[2]*p2c[2]) / (p2c_l * m2c_l)
            start_id = 0
            for l in range(1, l_limit):
                l_result = calc_legendre(l_list, cos_gamma, l+1, start_id)
                pseudo_result += l_result*(2*l+1)*(rr**l)*part_value[pid]
                start_id += l+1

    pseudo_value[i] = pseudo_result/number_makino
    


@annotate(
    int="i, level, l_limit, number_makino",
    doublep="pseudo_value, pseudo_x, pseudo_y, pseudo_z, part_value, part_x, part_y, part_z, cx, cy, cz, l_list",
    length="double"
)
def calc_pseudoparticles(
    i, pseudo_value, pseudo_x, pseudo_y, pseudo_z, part_value, 
    part_x, part_y, part_z, cx, cy, cz, 
    number_makino, length, level, l_limit, l_list
):
    j, l, cid, pid, start_id = declare("int", 5)
    p2c, m2c = declare("matrix(3)", 2)
    m2c_l, p2c_l, cos_gamma, rr, l_result, pseudo_result = declare("double", 6)
    cid = cast(floor(i / number_makino), "int")
    m2c[0] = pseudo_x[i] - cx[cid]
    m2c[1] = pseudo_y[i] - cx[cid]
    m2c[2] = pseudo_z[i] - cx[cid]
    m2c_l = length/(2**(level+1))*3
    pseudo_result = 0
    pid = (cid << 3)*number_makino
    for j in range(8*number_makino):
        p2c[0] = part_x[pid+j] - cx[cid]
        p2c[1] = part_y[pid+j] - cy[cid]
        p2c[2] = part_z[pid+j] - cz[cid]
        p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
        pseudo_result += part_value[pid+j]
        if p2c_l != 0:
            rr = p2c_l / m2c_l
            cos_gamma = (m2c[0]*p2c[0] + m2c[1]*p2c[1] + m2c[2]*p2c[2]) / (p2c_l * m2c_l)
            start_id = 0
            for l in range(1, l_limit):
                l_result = calc_legendre(l_list, cos_gamma, l+1, start_id)
                pseudo_result += l_result*(2*l+1)*(rr**l)*part_value[pid+j]
                start_id += l+1

    pseudo_value[i] = pseudo_result/number_makino


# need to do this for all levels 1-finest (both included) (need Vb for level 2 as well)
@annotate(
    i="int", associate_ids="intp", level="intp",
    doublep="cx, cy, cz", double="length, x_min, y_min, z_min"
)
def find_associates(
    i, cx, cy, cz, associate_ids, level, 
    length, x_min=0, y_min=0, z_min=0
):
    ax, ay, az, dist_offset = declare("double", 4)
    b_len, j, k, l, count, count_reg = declare("int", 6)
    b_len = 2**level[i]
    dist_offset = length/b_len
    count = 0
    count_reg = 0
    for j in range(-1, 2, 1):
        for k in range(-1, 2, 1):
            for l in range(-1, 2, 1):
                if count != 13 or count_reg == 1:
                    ax = cx[i] + j*dist_offset
                    ay = cy[i] + k*dist_offset
                    az = cz[i] + l*dist_offset
                    if ((ax < length) and (ay < length) and (az < length) and (ax > 0) and (ay > 0) and (az > 0)):
                        associate_ids[26*i + count] = get_cell_id(ax, ay, az, b_len, x_min, y_min, z_min, length)
                    else:
                        associate_ids[26*i + count] = -1
                    count += 1
                else:
                    count_reg = 1


@annotate(
    double="part_value, part_x, part_y, part_z, point_x, point_y, point_z", 
    return_="double"
)
def direct_computation(
    part_value, part_x, part_y, part_z, point_x, point_y, point_z
):
    value, dist = declare("double", 2)
    value = 0
    dist = sqrt((part_x-point_x)**2+(part_y-point_y)**2+(part_z-point_z)**2)
    value += part_value/dist

    return value


@annotate(double="cx, cy, cz, ax, ay, az, cell_radius", return_="int")
def is_well_separated(cx, cy, cz, ax, ay, az, cell_radius):
    dist = declare("double", 2)
    dist = sqrt((cx-ax)**2+(cy-ay)**2+(cz-az)**2)
    if (dist - 3*cell_radius) >= 0:
        return 1
    else:
        return 0


# all arrays of level at which being calculated, associates id parent level
@annotate(
    doublep="inner_value, inner_x, inner_y, inner_z, outer_value, outer_x, outer_y, outer_z, cx, cy, cz",
    int="i, level, number_makino", associate_ids="intp", length="double"
)
def local_coeff(
    i, inner_value, inner_x, inner_y, inner_z, 
    outer_value, outer_x, outer_y, outer_z, 
    cx, cy, cz, associate_ids, number_makino, 
    level, length
):
    cell_id, parent_id, a_id, child_id = declare("int", 4)
    j, k, n = declare("int", 3)
    cell_radius = declare("double")
    cell_radius = sqrt(3)*length/(2**(level+1))
    cell_id = cast(floor(i / number_makino), "int")
    parent_id = cell_id >> 3
    for j in range(26):
        a_id = associate_ids[parent_id*26+j]
        if a_id != -1:
            child_id = a_id << 3
            for k in range(8):
                if (is_well_separated(cx[cell_id], cy[cell_id], cz[cell_id], cx[child_id], cy[child_id], cz[child_id], cell_radius) == 1):
                    for n in range(number_makino):
                        inner_value[i] += direct_computation(outer_value[child_id*number_makino+n], outer_x[child_id*number_makino+n], outer_y[child_id*number_makino+n], outer_z[child_id*number_makino+n], inner_x[i], inner_y[i], inner_z[i])
                child_id += 1

@annotate(
    doublep="inner_value, inner_x, inner_y, inner_z, l_list",
    double="cx, cy, cz, point_x, point_y, point_z, length",
    int="number_makino, level, inner_start_id, l_limit",
    return_="double"
)
def local_expansion(
    inner_value, inner_x, inner_y, inner_z, cx, cy, cz,
    point_x, point_y, point_z, number_makino, level, 
    length, inner_start_id, l_list, l_limit
):
    j, l, start_inner, start_id = declare("int", 4)
    result, p2c_l, cos_gamma, rr, i2c_l, l_result = declare("double", 6)
    p2c, i2c = declare("matrix(3)", 2)
    p2c[0] = point_x - cx
    p2c[1] = point_y - cy
    p2c[2] = point_z - cz
    p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
    result = 0
    i2c_l = 0.5*length/(2**(level+1))
    for j in range(number_makino):
        start_inner = inner_start_id + j
        i2c[0] = inner_x[start_inner] - cx
        i2c[1] = inner_y[start_inner] - cy
        i2c[2] = inner_z[start_inner] - cz
        result += inner_value[start_inner]
        if p2c_l != 0:
            cos_gamma = (i2c[0]*p2c[0] + i2c[1]*p2c[1] + i2c[2]*p2c[2]) / (p2c_l * i2c_l)
            rr = p2c_l / i2c_l
            start_id = 0
            for l in range(1, l_limit):
                l_result = calc_legendre(l_list, cos_gamma, l+1, start_id)
                result += l_result*(2*l+1)*(rr**l)*inner_value[start_inner]
                start_id += l+1
        
    return result/number_makino

# give parent wala level
@annotate(
    int="i, level, number_makino, l_limit",
    doublep="innerc_value, innerc_x, innerc_y, innerc_z, px, py, pz, innerp_value, innerp_x, innerp_y, innerp_z, l_list",
    length="double"
)
def transfer_local(
    i, innerc_value, innerc_x, innerc_y, innerc_z, px, py, pz, 
    innerp_value, innerp_x, innerp_y, innerp_z, level, length, 
    number_makino, l_list, l_limit
):
    pid, cid = declare("int", 2)
    cid = cast(floor(i / number_makino), "int")
    pid = cid >> 3
    innerc_value[i] += local_expansion(innerp_value, innerp_x, innerp_y, innerp_z, px[pid], py[pid], pz[pid], innerc_x[i], innerc_y[i], innerc_z[i], number_makino, level, length, pid*number_makino, l_list, l_limit)


@annotate(
    int="i, level, number_makino, l_limit",
    doublep="cx, cy, cz, inner_value, inner_x, inner_y, inner_z, l_list, part_x, part_y, part_z, part_value, part_result",
    length="double",
    intp="bin_count, indices, start, associate_ids"
)
def compute_value(
    i, cx, cy, cz, inner_value, inner_x, inner_y, inner_z, associate_ids, 
    number_makino, level, length, l_list, l_limit, part_x, part_y, part_z, 
    part_value, bin_count, start, indices, part_result
):
    j, k, l, m, aid, pid, p2id, p3id = declare("int", 8)
    for j in range(bin_count[i]):
        pid = indices[start[i] + j]
        for k in range(26):
            aid = associate_ids[i*26 + k]
            if aid != -1:
                for l in range(bin_count[aid]):
                    p2id = indices[start[aid] + l]
                    part_result[pid] += direct_computation(part_value[p2id], part_x[p2id], part_y[p2id], part_z[p2id], part_x[pid], part_y[pid], part_z[pid])
        
        for m in range(bin_count[i]):
            p3id = indices[start[i] + m]
            if p3id != pid:
                part_result[pid] += direct_computation(part_value[p3id], part_x[p3id], part_y[p3id], part_z[p3id], part_x[pid], part_y[pid], part_z[pid])

        part_result[pid] += local_expansion(inner_value, inner_x, inner_y, inner_z, cx[i], cy[i], cz[i], part_x[pid], part_y[pid], part_z[pid], number_makino, level, length, i*number_makino, l_list, l_limit)


@annotate(int="i, n", doublep="value, x, y, z, result")
def direct_solve(i, value, x, y, z, result, n):
    j = declare("int")
    for j in range(n):
        if i != j:
            result[i] += direct_computation(value[j], x[j], y[j], z[j], x[i], y[i], z[i])

## testing part over here

backend = 'cython'
# get_config().use_openmp = True

n = 2
number_makino = 4
np.random.seed(0)
rnd = np.random.random((4, n))
x = rnd[0]
y = rnd[1]
z = rnd[2]
# prop = rnd[3]
prop = np.ones(n)

# x = np.array([0.03125, 0.03, 0.09375])
# y = np.array([0.03125, 0.03, 0.09375])
# z = np.array([0.03125, 0.03, 0.09375])
# prop = np.array([1, 1, 1], dtype=np.float64)

# x = np.array([0.0312, 0.0937, 0.9687, 0.9062])
# y = np.array([0.0312, 0.0312, 0.9687, 0.9687])
# z = np.array([0.0312, 0.0312, 0.9687, 0.9687])
# prop = np.ones(n)

# x = np.array([0.03125, 0.03])
# y = np.array([0.03125, 0.03])
# z = np.array([0.03125, 0.03])
# x = np.array([0.125, 0.875])
# y = np.array([0.125, 0.875])
# z = np.array([0.125, 0.875])
# prop = np.array([1, 1], dtype=np.float64)

level = 4
length = 1
x_min = 0
y_min = 0
z_min = 0
b_len = 2 ** level

total_blocks = int(64 * (8 ** (level - 1) - 1) / 7)

wasteblocks = 8 ** (level + 1) * int((8 ** (8 - level) - 1) / 7)
npzfile = np.load("centers.npz")
cx = npzfile["cx"]
cy = npzfile["cy"]
cz = npzfile["cz"]
level_list = npzfile["level_matrix"]
level_list = level_list.astype("int32")

cx = cx[wasteblocks:]
cy = cy[wasteblocks:]
cz = cz[wasteblocks:]
level_list = level_list[wasteblocks:]

outer_value = np.zeros(total_blocks*number_makino)
outer_x = np.zeros(total_blocks*number_makino)
outer_y = np.zeros(total_blocks*number_makino)
outer_z = np.zeros(total_blocks*number_makino)

inner_value = np.zeros(total_blocks*number_makino)
inner_x = np.zeros(total_blocks*number_makino)
inner_y = np.zeros(total_blocks*number_makino)
inner_z = np.zeros(total_blocks*number_makino)

associate_ids = np.zeros((total_blocks+8)*26, dtype=np.int32)

sph_points, order = spherical_points(number_makino)
sph_points = sph_points.astype(np.float64)
l_limit = int(order/2)+1
size_l_list = int(order*(order+1)/2) - 1
l_list = np.zeros(size_l_list)

count = 0
for i in range(1, order):
    temp_list = np.array(legendre(i))
    l_list[count:count+i+1] = temp_list[::-1]

    count += i+1


bin_count = np.zeros(8 ** level, dtype=np.int32)
start = np.zeros_like(bin_count, dtype=np.int32)
bin_offset = np.zeros_like(x, dtype=np.int32)
indices = np.zeros_like(x, dtype=np.int32)

count = 0
sb = np.zeros(level+1, dtype=np.int32)
for l in range(level, -1, -1):
    sb[l] = count
    count += 8**l

sb = sb[::-1]

result = np.zeros_like(x)
direct_result = np.zeros_like(x)

einitial_bin_count = Elementwise(initial_bin_count, backend=backend)
eget_bin_count = Elementwise(get_bin_count, backend=backend)
cum_bin_count = Scan(
    input_bin_count, output_bin_count, "a+b", dtype=np.int32, 
    backend=backend
)
estart_indices = Elementwise(start_indices, backend=backend)
esetting_pseudoparticles = Elementwise(setting_pseudoparticles, backend=backend)
ecalc_pseudoparticles_fine = Elementwise(calc_pseudoparticles_fine, backend=backend)
ecalc_pseudoparticles = Elementwise(calc_pseudoparticles, backend=backend)
efind_associates = Elementwise(find_associates, backend=backend)
elocal_coeff = Elementwise(local_coeff, backend=backend)
etransfer_local = Elementwise(transfer_local, backend=backend)
ecompute_value = Elementwise(compute_value, backend=backend)
edirect_solve = Elementwise(direct_solve, backend=backend)


einitial_bin_count(bin_count)
eget_bin_count(x, y, z, b_len, bin_count, bin_offset)
cum_bin_count(bin_count=bin_count, start_index=start)
estart_indices(x, y, z, b_len, bin_offset, start, indices)

esetting_pseudoparticles(
    cx[:-8], cy[:-8], cz[:-8], outer_x, outer_y, outer_z, 
    inner_x, inner_y, inner_z, sph_points, length, 
    level_list[:-8], number_makino
)

efind_associates(cx, cy, cz, associate_ids, level_list, length, x_min, y_min, z_min)

s0 = sb[0]
s1 = sb[1]
m0 = s0*number_makino
m1 = s1*number_makino


ecalc_pseudoparticles_fine(
    outer_value[m0:m1], outer_x[m0:m1], outer_y[m0:m1], outer_z[m0:m1], 
    prop, x, y, z, cx[s0:s1], cy[s0:s1], cz[s0:s1], indices, start, 
    bin_count, number_makino, length, level, l_limit, l_list
)

for l in range(level-1, 1, -1):
    s0 = sb[level-l-1]
    s1 = sb[level-l]
    s2 = sb[level-l+1]
    t0 = s0*number_makino
    t1 = s1*number_makino
    t2 = s2*number_makino
    

    ecalc_pseudoparticles(
        outer_value[t1:t2], outer_x[t1:t2], outer_y[t1:t2], outer_z[t1:t2], 
        outer_value[t0:t1], outer_x[t0:t1], outer_y[t0:t1], outer_z[t0:t1], 
        cx[s1:s2], cy[s1:s2], cz[s1:s2], number_makino, length, l, 
        l_limit, l_list
    )


# s0-s1 real level, s1-s2 parent level
for l in range(2, level+1):
    s0 = sb[level-l]
    s1 = sb[level-l+1]
    s2 = sb[level-l+2]
    m0 = s0*number_makino
    m1 = s1*number_makino
    a1 = s1*26
    a2 = s2*26

    elocal_coeff(
        inner_value[m0:m1], inner_x[m0:m1], inner_y[m0:m1], inner_z[m0:m1], 
        outer_value[m0:m1], outer_x[m0:m1], outer_y[m0:m1], outer_z[m0:m1], 
        cx[s0:s1], cy[s0:s1], cz[s0:s1], associate_ids[a1:a2], number_makino, 
        l, length
    )


# # s0-s1 child level s1-s2 real level
for l in range(2, level):
    s0 = sb[level-l-1]
    s1 = sb[level-l]
    s2 = sb[level-l+1]
    m0 = s0*number_makino
    m1 = s1*number_makino
    m2 = s2*number_makino

    etransfer_local(
        inner_value[m0:m1], inner_x[m0:m1], inner_y[m0:m1], inner_z[m0:m1], 
        cx[s1:s2], cy[s1:s2], cz[s1:s2], inner_value[m1:m2], inner_x[m1:m2], 
        inner_y[m1:m2], inner_z[m1:m2], l, length, 
        number_makino, l_list, l_limit
    )


s0 = sb[0]
s1 = sb[1]
m0 = s0*number_makino
m1 = s1*number_makino
a0 = s0*26
a1 = s1*26

ecompute_value(
    cx[s0:s1], cy[s0:s1], cz[s0:s1], inner_value[m0:m1], inner_x[m0:m1], 
    inner_y[m0:m1], inner_z[m0:m1], associate_ids[a0:a1], number_makino, 
    level, length, l_list, l_limit, x, y, z, prop, 
    bin_count, start, indices, result
)

edirect_solve(prop, x, y, z, direct_result, n)


# print(np.nonzero(bin_count))

print(direct_result)
print(result)

# print(np.mean(np.abs(result-direct_result)))

# print(is_well_separated(cx[228+4096], cy[228+4096], cz[228+4096], cx[453+4096], cy[453+4096], cz[453+4096], 0.5*sqrt(3)*(1/2**3)))

# print(x)
# print(y)
# print(z)

# magic_number = 1826
# magic_number = 3625
# magic_number = 228 + 4096
# magic_number = 453 + 4096
# print(cx[magic_number], cy[magic_number], cz[magic_number])
# for i in inner_value[magic_number*number_makino:(magic_number+1)*number_makino]:
#     print(i)