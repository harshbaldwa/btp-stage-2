def find_u_w_list(b, cell, Ub, Wb):
    if not b.near_range(cell):
        Wb.append(cell)
    else:
        if cell.is_parent():
            for child in cell.children:
                find_u_w_list(b, child, Ub, Wb)
        else:
            Ub.append(cell)


def u_w_list(b, kind):
    Ub, Wb = [], []

    Ub.append(b)

    for near_cell in b.near_cells:
        if near_cell.is_parent():
            find_u_w_list(b, near_cell, Ub, Wb)
        else:
            Ub.append(near_cell)

    for particle in b.particles:
        for cell in Ub:
            particle.pot += kind.direct(cell.particles, 
                                          particle.position)
            # if all(b.center == [0.125, 0.625]):
            #     print("Direct - {} - {}".format(cell.center, kind.direct(cell.particles, particle.position)))

        for cell in Wb:
            particle.pot += kind.direct(cell.inner_equi,
                                          particle.position)
            # if all(b.center == [0.125, 0.625]):
            #     print("Multipole - {} - {}".format(cell.center, kind.direct(cell.inner_equi, particle.position)))


def v_list(b, kind):
    for near_cell in b.parent.near_cells:
        if near_cell.level == b.parent.level:
            if near_cell.is_parent():
                for child in near_cell.children:
                    if not b.near_range(child):
                        kind.M2L_check(child, b)



def z_list(b, kind):
    for near_cell in b.parent.near_cells:
        if not near_cell.is_parent():
            if not b.near_range(near_cell):
                kind.S2L_check(near_cell, b)


def equivalent_calculations(tree, kind):
    for cell in tree.childless:
        kind.S2M(cell)

    for level in range(tree.depth - 1, 1, -1):
        for cell in tree.cells[level]:
            for child in cell.children:
                kind.M2M_check(child)
            kind.M2M_equi(cell)


def compute_local_expasions(tree, kind):

    depth = tree.depth

    for level in range(2, depth + 1):
        for cell in tree.cells[level]:
            v_list(cell, kind)
            z_list(cell, kind)

    for level in range(2, depth):
        for cell in tree.cells[level]:
            kind.X2L_equi(cell)
            kind.L2L_check(cell)


def compute_value(tree, kind):

    equivalent_calculations(tree, kind)
    compute_local_expasions(tree, kind)

    for cell in tree.childless:
        u_w_list(cell, kind)
        kind.X2L_equi(cell)
        for particle in cell.particles:
            particle.pot += kind.direct(cell.outer_equi,
                                     particle.position)
            # if all(cell.center == [0.125, 0.625]):
            #     print("Local - {} - {}".format(cell.center, kind.direct(cell.outer_equi, particle.position)))