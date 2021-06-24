from compyle.api import annotate, wrap, Elementwise, Scan, get_config
from compyle.types import declare
from math import sqrt
import numpy as np

@annotate(i="int", gdoublep="i_harsh, o_harsh")
def testest(i, i_harsh, o_harsh):
    i_harsh[i] = o_harsh[i]**2

backend = 'cython'

n = 1000
i_harsh = np.ones(n)*1e-5
o_harsh = np.zeros(n)

i_harsh, o_harsh = wrap(i_harsh, o_harsh, backend=backend)

etestest = Elementwise(testest, backend=backend)

etestest(i_harsh, o_harsh)

print(i_harsh[0], o_harsh[0])

