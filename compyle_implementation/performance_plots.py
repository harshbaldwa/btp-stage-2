from compyle.api import get_config
from main import solver
from math import log

# N = [4000, 5000, 8000, 10000, 16000, 20000, 25000, 32000, 40000, 50000, 64000, 100000, 128000]
N = [150000, 256000]
seed = 0
p = 10
backend = 'cython'
get_config().use_openmp = True
Error = []
Time_f = []
Time_d = []
Speedup = []


for i in range(len(N)):
    n = N[i]
    # l = int(log(n)/log(4)) + 1
    l = 5
    error, time_f, time_d = solver(n, l, p, seed, backend)
    Error.append(error)
    Time_f.append(time_f)
    Time_d.append(time_d)
    Speedup.append(time_d/time_f)
    print("N - {}\nSpeedup - {}\nError - {}\nTime(fmm) - {}\n".format(n, time_d/time_f, error,
                                                     time_f
                                                     ))