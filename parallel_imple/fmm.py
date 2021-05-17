import math

def nCr(n, r):
    if n >= r:
        f = math.factorial
        return f(n) // (f(r) * f(n-r))
    else:
        return 0


def log_complex(z):
    result = math.log(abs(z)) + math.atan2(z.imag, z.real)
    return result


def calculate_multipole_fine(start, indices, bin_count, x, y, charge, multipole, p, cx, cy):
    for i in range(len(start)):
        for j in range(bin_count[i]):
            pid = indices[start[i]+j]
            multipole[(p+1)*i] += charge[pid]
            for k in range(1, p+1):
                multipole[(p+1)*i+k] += -(charge[pid]/k)*complex(x[pid]-cx[i], y[pid]-cy)**k


def transfer_multipole(multipole_parent, multipole, p, cx_parent, cx, cy_parent, cy):
    for i in range(len(multipole_parent)):
        for j in range(4):
            bid_base = 4*i + j*(p+1)
            l = i%(p+1)
            if l == 0:
                multipole_parent[i] += multipole[bid_base]
            else:
                for k in range(1, l+1):
                    multipole_parent[i] += multipole[bid_base + k]*complex(cx[bid_base + k] - cx_parent[i], cy[bid_base + k] - cy_parent[i])**(l-k)*nCr(l-1, k-1) - multipole[bid_base]*complex(cx[bid_base + k] - cx_parent[i], cy[bid_base + k] - cy_parent[i])**l/l


def direct_computation(particles, particles_x, particles_y, direct_charge, direct_x, direct_y):
    for i in range(len(particles)):
        for j in range(len(direct_charge)):
            particles[i] += direct_charge[j]*log_complex(complex(particles_x[i] - direct_x[j], particles_y[i] - direct_y[j]))


def multipole_expansion(particles, particles_x, particles_y, multipole, cx, cy, p):
    for i in range(len(particles)):
        for j in range(len(multipole)):
            l = j%(p+1)
            if l == 0:
                particles[i] += multipole[j]*log_complex(complex(particles_x[i] - cx[j], particles_y[i] - cy[j]))
            else:
                particles[i] += multipole[j] / (complex(particles_x[i] - cx[j], particles_y[i] - cy[j]))**l

