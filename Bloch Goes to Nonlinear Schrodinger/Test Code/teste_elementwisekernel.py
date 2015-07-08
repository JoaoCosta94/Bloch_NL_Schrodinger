import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la
from pyopencl.elementwise import ElementwiseKernel
import matplotlib.pyplot as plt

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

X_h = np.array([1+1j*1, 2+1j*2, 3+1j*3]).astype(np.complex64)
Y_h = np.array([4+1j*4, 5+1j*5, 6+1j*6]).astype(np.complex64)
dados_h = []
for i in range(3):
    dados_h.append( np.array([X_h[0], X_h[1], X_h[2], Y_h[0], Y_h[1], Y_h[2]]).astype(np.complex64) )

dados_h = np.array(dados_h).astype(np.complex64)
print dados_h

dados_d = cl_array.to_device(queue, dados_h)
res_d = cl_array.to_device(queue, np.empty_like(X_h))

teste = ElementwiseKernel(ctx,
        "float8 *dados, "
        "float2 *res ",
        """
        res[i] = dados[i].s5 
        """,
        "teste")

teste(dados_d, res_d)
print res_d
