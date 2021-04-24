import math


def nCr(n, r):
        if n >= r:
            f = math.factorial
            return f(n) // (f(r) * f(n-r))
        else:
            return 0


def calculate_multipole(start, indices, bin_count, x, y, charge, multipole, p, cx, cy):
    for i in range(len(start)):
        for j in range(bin_count[i]):
            pid = indices[start[i]+j]
            multipole[(p+1)*i] += charge[pid]
            for k in range(1, p):
                multipole[(p+1)*i+k] += -(charge[pid]/k)*complex(x[pid]-cx[i], y[pid]-cy)**k


def transfer_multipole(multipole_parent, multipole, p, cx_parent, cx, cy_parent, cy):
    for i in range(len(multipole_parent)):
        for j in range(4):
            if i%(p+1) == 0:
                multipole_parent[i] += multipole[4*i + j*(p+1)]
            else:
                for k in range(1, i%(p+1)):
                    multipole_parent[i] += multipole[4*i + j*(p+1) + k]*complex(cx[4*i + j*(p+1) + k] - cx_parent[i], cy[4*i + j*(p+1) + k] - cy_parent[i])**(i%(p+1)-k)*nCr(i%(p+1)-1, k-1) - multipole[4*i + j*(p+1)]*complex(cx[4*i + j*(p+1) + k] - cx_parent[i], cy[4*i + j*(p+1) + k] - cy_parent[i])**(i%(p+1))/(i%(p+1))
