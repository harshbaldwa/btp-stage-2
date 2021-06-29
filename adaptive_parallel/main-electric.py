from compyle.api import annotate, wrap, Elementwise, Scan, get_config
from compyle.types import declare
from compyle.low_level import cast, atomic_inc
from compyle.parallel import serial
from math import sqrt, floor, log

from spherical_points import spherical_points
from scipy.special import legendre
import numpy as np
import time
import argparse


@annotate(double="x, y, z, x_min, y_min, z_min, length, b_len", return_="int")
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


@annotate(i='int', bin_count='gintp')
def initial_bin_count(i, bin_count):
    bin_count[i] = 0


@serial
@annotate(i="int", gintp="bin_count, bin_offset", gdoublep="x, y, z", b_len="double")
def get_bin_count(i, x, y, z, b_len, bin_count, bin_offset):
    id = declare("int")
    idx = declare("int")
    id = get_cell_id(x[i], y[i], z[i], b_len, 0, 0, 0, 1)
    idx = atomic_inc(bin_count[id])
    bin_offset[i] = idx


@annotate(i="int", gintp="bin_p, bin_c")
def transfer_bin_count(i, bin_p, bin_c):
    j = declare("int")
    for j in range(8):
        bin_p[i] += bin_c[8*i+j]


@annotate(i="int", bin_count="gintp", return_="int")
def input_bin_count(i, bin_count):
    if i == 0:
        return 0
    else:
        return bin_count[i - 1]


@annotate(int="i, item", start_index="gintp")
def output_bin_count(i, item, start_index):
    start_index[i] = item


@annotate(int="i, start_term", start="gintp")
def start_manipulation(i, start, start_term):
    start[i] -= start_term


@annotate(
    i="int",
    b_len="double",
    intp="bin_offset, indices, start_index",
    gdoublep="x, y, z"
)
def start_indices(i, x, y, z, b_len, bin_offset, start_index, indices):
    id = declare("int")
    id = get_cell_id(x[i], y[i], z[i], b_len, 0, 0, 0, 1)
    indices[start_index[id] + bin_offset[i]] = i


@annotate(
    int="i, number_makino",
    gdoublep="cx, cy, cz, outer_x, outer_y, outer_z, inner_x, inner_y, inner_z, sph_points",
    level="gintp",
    length="double"
)
def setting_pseudoparticles(
    i, cx, cy, cz, outer_x, outer_y, outer_z, inner_x, inner_y, inner_z, 
    sph_points, length, level, number_makino
):
    j = declare("int")
    size_cell = declare("double")
    size_cell = length/(2.0**(level[i]+1))
    for j in range(number_makino):
        outer_x[i*number_makino + j] = cx[i] + sph_points[3*j]*3*size_cell
        outer_y[i*number_makino + j] = cy[i] + sph_points[3*j+1]*3*size_cell
        outer_z[i*number_makino + j] = cz[i] + sph_points[3*j+2]*3*size_cell
        inner_x[i*number_makino + j] = cx[i] + sph_points[3*j]*0.5*size_cell
        inner_y[i*number_makino + j] = cy[i] + sph_points[3*j+1]*0.5*size_cell
        inner_z[i*number_makino + j] = cz[i] + sph_points[3*j+2]*0.5*size_cell


@annotate(int="l_len, start_id", cos_gamma="double", l_list="gdoublep", return_="double")
def calc_legendre(l_list, cos_gamma, l_len, start_id):
    i = declare("int")
    result = declare("double")
    result = 0
    for i in range(l_len):
        result += l_list[start_id+i]*(cos_gamma**i)

    return result


@annotate(
    int="i, level, l_limit, number_makino",
    gintp="bin_count, indices, start",
    gdoublep="pseudo_value, pseudo_x, pseudo_y, pseudo_z, part_value, part_x, part_y, part_z, cx, cy, cz, l_list",
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
    cid = cast(floor(i*1.0 / number_makino), "int")
    m2c[0] = pseudo_x[i] - cx[cid]
    m2c[1] = pseudo_y[i] - cy[cid]
    m2c[2] = pseudo_z[i] - cz[cid]
    m2c_l = length/(2.0**(level+1))*3
    pseudo_result = 0
    pseudo_value[i] = 0
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
    gdoublep="pseudo_value, pseudo_x, pseudo_y, pseudo_z, part_value, part_x, part_y, part_z, cx, cy, cz, l_list",
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
    cid = cast(floor(i*1.0 / number_makino), "int")
    m2c[0] = pseudo_x[i] - cx[cid]
    m2c[1] = pseudo_y[i] - cy[cid]
    m2c[2] = pseudo_z[i] - cz[cid]
    m2c_l = length/(2.0**(level+1))*3
    pseudo_result = 0
    pseudo_value[i] = 0
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


@annotate(
    i="int", gintp="associate_ids, level",
    gdoublep="cx, cy, cz", double="length, x_min, y_min, z_min"
)
def find_associates(
    i, cx, cy, cz, associate_ids, level, 
    length, x_min=0, y_min=0, z_min=0
):
    ax, ay, az, dist_offset, b_len = declare("double", 5)
    j, k, l, count, count_reg = declare("int", 5)
    b_len = 2.0**level[i]
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


@annotate(double="cx, cy, cz, ax, ay, az, cell_dist", return_="int")
def is_well_separated(cx, cy, cz, ax, ay, az, cell_dist):
    dist = declare("double", 2)
    dist = sqrt((cx-ax)**2+(cy-ay)**2+(cz-az)**2)
    if (dist - cell_dist) >= 0:
        return 1
    else:
        return 0


@annotate(double="cx, cy, cz, ax, ay, az, cell_dist", return_="int")
def is_adjacent(cx, cy, cz, ax, ay, az, cell_dist):
    dist = declare("double", 2)
    dist = sqrt((cx-ax)**2+(cy-ay)**2+(cz-az)**2)
    if (dist - cell_dist) <= 0:
        return 1
    else:
        return 0


@annotate(
    int="i, number_makino, pid_start",
    gdoublep="inner_value, inner_x, inner_y, inner_z, outer_value, outer_x, outer_y, outer_z"
)
def calc_v_list(i, number_makino, pid_start, inner_value, inner_x, inner_y, inner_z, outer_value, outer_x, outer_y, outer_z):
    n, pid = declare("int", 2)
    for n in range(number_makino):
        pid = pid_start + n
        inner_value[i] += direct_computation(outer_value[pid], outer_x[pid], outer_y[pid], outer_z[pid], inner_x[i], inner_y[i], inner_z[i])


@annotate(
    int="i, num_loop, pid_start",
    gdoublep="inner_value, inner_x, inner_y, inner_z, part_value, part_x, part_y, part_z",
    indices="gintp"
)
def calc_z_list(i, num_loop, pid_start, inner_value, inner_x, inner_y, inner_z, part_value, part_x, part_y, part_z, indices):
    n, pid = declare("int", 2)
    for n in range(num_loop):
        pid = indices[pid_start + n]
        inner_value[i] += direct_computation(part_value[pid], part_x[pid], part_y[pid], part_z[pid], inner_x[i], inner_y[i], inner_z[i])

@annotate(
    gdoublep="inner_value, inner_x, inner_y, inner_z, outer_value, outer_x, outer_y, outer_z, cx, cy, cz, part_value, part_x, part_y, part_z",
    int="i, level, number_makino", gintp="associate_ids, bin_count, indices, start", length="double"
)
def local_coeff(
    i, inner_value, inner_x, inner_y, inner_z, 
    outer_value, outer_x, outer_y, outer_z, 
    cx, cy, cz, associate_ids, number_makino, 
    level, length, 
    bin_count, indices, start, 
    part_value, part_x, part_y, part_z
):
    cell_id, parent_id, a_id, child_id = declare("int", 4)
    j, k = declare("int", 4)
    cell_radius = declare("double")
    cell_radius = sqrt(3.0)*length/(2.0**(level+1))
    cell_id = cast(floor(i*1.0 / number_makino), "int")
    parent_id = cell_id >> 3
    inner_value[i] = 0
    for j in range(26):
        a_id = associate_ids[parent_id*26+j]
        if a_id != -1:
            child_id = a_id << 3
            for k in range(8):
                if (is_well_separated(cx[cell_id], cy[cell_id], cz[cell_id], cx[child_id], cy[child_id], cz[child_id], 3*cell_radius) == 1):
                    calc_v_list(i, number_makino, child_id*number_makino, inner_value, inner_x, inner_y, inner_z, outer_value, outer_x, outer_y, outer_z)
                else:
                    if (is_adjacent(cx[cell_id], cy[cell_id], cz[cell_id], cx[child_id], cy[child_id], cz[child_id], 2*cell_radius) == 0):
                        calc_z_list(i, bin_count[child_id], start[child_id], inner_value, inner_x, inner_y, inner_z, part_value, part_x, part_y, part_z, indices)
                child_id += 1


@annotate(
    gdoublep="inner_value, inner_x, inner_y, inner_z, l_list",
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
    i2c_l = 0.5*length/(2.0**(level+1))
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

@annotate(
    int="i, level, number_makino, l_limit",
    gdoublep="innerc_value, innerc_x, innerc_y, innerc_z, px, py, pz, innerp_value, innerp_x, innerp_y, innerp_z, l_list",
    length="double"
)
def transfer_local(
    i, innerc_value, innerc_x, innerc_y, innerc_z, px, py, pz, 
    innerp_value, innerp_x, innerp_y, innerp_z, level, length, 
    number_makino, l_list, l_limit
):
    pid, cid = declare("int", 2)
    cid = cast(floor(i*1.0 / number_makino), "int")
    pid = cid >> 3
    innerc_value[i] += local_expansion(innerp_value, innerp_x, innerp_y, innerp_z, px[pid], py[pid], pz[pid], innerc_x[i], innerc_y[i], innerc_z[i], number_makino, level, length, pid*number_makino, l_list, l_limit)



@annotate(
    int="i, level, number_makino, l_limit",
    gdoublep="cx, cy, cz, inner_value, inner_x, inner_y, inner_z, l_list, part_x, part_y, part_z, part_value, part_result",
    length="double",
    gintp="bin_count, indices, start, associate_ids"
)
def compute_value(
    i, cx, cy, cz, inner_value, inner_x, inner_y, inner_z, associate_ids, 
    number_makino, level, length, l_list, l_limit, part_x, part_y, part_z, 
    part_value, bin_count, start, indices, part_result
):
    j, k, l, m, associate_num, aid, pid, p2id, p3id = declare("int", 9)
    associate_num = 26
    for j in range(bin_count[i]):
        pid = indices[start[i] + j]
        part_result[pid] = 0
        for k in range(associate_num):
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


@annotate(int="i, n_part", gdoublep="value, x, y, z, result")
def direct_solve(i, value, x, y, z, result, n_part):
    j = declare("int")
    result[i] = 0
    for j in range(n_part):
        if i != j:
            result[i] += direct_computation(value[j], x[j], y[j], z[j], x[i], y[i], z[i])



def solver(n, number_makino, level, compare_direct, compare_parallel, backend='cython'):

    LEVEL_MAX = 8

    np.random.seed(0)
    rnd = np.random.random((n, 3))
    x = rnd[:, 0]
    y = rnd[:, 1]
    z = rnd[:, 2]

    prop = np.ones(n)

    length = 1
    x_min = 0
    y_min = 0
    z_min = 0
    b_len = 2 ** level

    total_blocks = int(64 * (8 ** (level - 1) - 1) / 7)

    wasteblocks = 8 ** (level + 1) * int((8 ** (LEVEL_MAX - level) - 1) / 7)
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

    associate_ids = np.ones((total_blocks+8)*26, dtype=np.int32)*-1

    sph_points, order = spherical_points(number_makino)
    sph_points = sph_points.astype(np.float64)
    l_limit = int(order/2)+1
    size_l_list = int(l_limit*(l_limit+1)/2) - 1
    l_list = np.zeros(size_l_list)

    count = 0
    for i in range(1, l_limit):
        temp_list = np.array(legendre(i))
        l_list[count:count+i+1] = temp_list[::-1]

        count += i+1


    bin_count = np.zeros(total_blocks, dtype=np.int32)
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
    etransfer_bin_count = Elementwise(transfer_bin_count, backend=backend)
    cum_bin_count = Scan(
        input_bin_count, output_bin_count, "a+b", dtype=np.int32, 
        backend=backend
    )
    estart_manipulation = Elementwise(start_manipulation, backend=backend)
    estart_indices = Elementwise(start_indices, backend=backend)
    esetting_pseudoparticles = Elementwise(setting_pseudoparticles, backend=backend)
    ecalc_pseudoparticles_fine = Elementwise(calc_pseudoparticles_fine, backend=backend)
    ecalc_pseudoparticles = Elementwise(calc_pseudoparticles, backend=backend)
    efind_associates = Elementwise(find_associates, backend=backend)
    elocal_coeff = Elementwise(local_coeff, backend=backend)
    etransfer_local = Elementwise(transfer_local, backend=backend)
    ecompute_value = Elementwise(compute_value, backend=backend)
    edirect_solve = Elementwise(direct_solve, backend=backend)


    t_time = 0
    t_count = 0

    if (t_time < 1):
        start_tree = time.time()

        if t_count == 0:
            x, y, z, prop, cx, cy, cz, level_list, outer_value, outer_x, outer_y, outer_z, inner_value, inner_x, inner_y, inner_z, associate_ids, sph_points, l_list, bin_count, start, bin_offset, indices, result, direct_result = wrap(x, y, z, prop, cx, cy, cz, level_list, outer_value, outer_x, outer_y, outer_z, inner_value, inner_x, inner_y, inner_z, associate_ids, sph_points, l_list, bin_count, start, bin_offset, indices, result, direct_result, backend=backend)
            efind_associates(cx, cy, cz, associate_ids, level_list, length, x_min, y_min, z_min)
        
        einitial_bin_count(bin_count)
        eget_bin_count(x, y, z, b_len, bin_count[sb[0]:sb[1]], bin_offset)

        for l in range(level-1, 1, -1):
            s0 = sb[level-l-1]
            s1 = sb[level-l]
            s2 = sb[level-l+1]

            etransfer_bin_count(bin_count[s1:s2], bin_count[s0:s1])

        cum_bin_count(bin_count=bin_count, start_index=start)

        for l in range(level-1, 1, -1):
            s1 = sb[level-l]
            s2 = sb[level-l+1]

            estart_manipulation(start[s1:s2], start[s1])


        estart_indices(x, y, z, b_len, bin_offset, start[sb[0]:sb[1]], indices)

        esetting_pseudoparticles(
            cx[:-8], cy[:-8], cz[:-8], outer_x, outer_y, outer_z, 
            inner_x, inner_y, inner_z, sph_points, length, 
            level_list[:-8], number_makino
        )


        s0 = sb[0]
        s1 = sb[1]
        m0 = s0*number_makino
        m1 = s1*number_makino


        ecalc_pseudoparticles_fine(
            outer_value[m0:m1], outer_x[m0:m1], outer_y[m0:m1], outer_z[m0:m1], 
            prop, x, y, z, cx[s0:s1], cy[s0:s1], cz[s0:s1], indices, start[s0:s1], 
            bin_count[s0:s1], number_makino, length, level, l_limit, l_list
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
                l, length, bin_count[s0:s1], indices, start[s0:s1], prop, x, y, z
            )


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
            bin_count[s0:s1], start[s0:s1], indices, result
        )

        end_tree = time.time()

        t_time += end_tree - start_tree
        t_count += 1

    t_time = t_time/t_count

    if compare_direct:
        
        d_time = 0
        d_count = 0

        if compare_parallel:
            if d_time < 0.5:
                start_direct = time.time()
                edirect_solve(prop, x, y, z, direct_result, n)
                end_direct = time.time()
                
                d_time += end_direct - start_direct
                d_count += 1

            if backend == 'opencl':
                direct_result.pull()
                direct_result = direct_result.data
        else:
            if backend == 'opencl':
                direct_result.pull()
                x.pull()
                y.pull()
                z.pull()
                prop.pull()
                direct_result = direct_result.data
                x = x.data
                y = y.data
                z = z.data
                prop = prop.data
            
            if d_time < 0.5:
                start_direct = time.time()
                for i in range(n):
                    direct_solve(i, prop, x, y, z, direct_result, n)
                end_direct = time.time()

                d_time += end_direct - start_direct
                d_count += 1

        d_time = d_time/d_count

    if backend == 'opencl':
        result.pull()
        result = result.data

    if compare_direct:
        return direct_result, result, d_time, t_time
    else:
        return [], result, 0, t_time


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of particles", type=int, default=10000)
    parser.add_argument("-l", "--level", help="depth of tree",
                        type=int, default=5)
    parser.add_argument("-p", help="number of pseudoparticles to use",
                        type=int, default=6)
    parser.add_argument("-b", "--backend", help="backend to use",
                        default='cython')
    parser.add_argument("-omp", "--openmp", help="use openmp for calculations",
                        action="store_true")
    parser.add_argument("-cd", "--compare_direct", help="whether to compare with direct",
                        action="store_true")                    
    parser.add_argument("-cp", "--compare_parallel", help="whether to compare speedup with serial or parallel direct",
                        action="store_true")

    args = parser.parse_args()

    if args.openmp:
        get_config().use_openmp = True

    direct_result, result, time_direct, time_tree = solver(args.n, args.p, args.level, args.compare_direct, args.compare_parallel, args.backend)

    print("Time taken by Anderson - ", time_tree)
    
    if args.compare_direct:
        print("Time taken by direct - ", time_direct)
        print("Speedup - ", time_direct/time_tree)
        print("Relative Error - ", np.mean(np.abs(result-direct_result)/direct_result))
