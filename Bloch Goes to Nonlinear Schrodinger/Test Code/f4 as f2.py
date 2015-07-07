import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la
from pyopencl.elementwise import ElementwiseKernel
import matplotlib.pyplot as plt

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

M = 3

X = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
Y = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
print X
print Y

DATA_h = []
for i in range(M):
    DATA_h.append(np.array([X[i], Y[i]]))
DATA_h = np.array(DATA_h)
print DATA_h

DATA_d = cl_array.to_device(queue, DATA_h)
RES_d = cl_array.to_device(queue, np.empty_like(DATA_h))

##f = open("f4_as_f2_kernel.cl", "r")
##Source = f.read()
##prg = cl.Program(ctx, Source).build()
##
##completeEvent = prg.soma(queue, (M,), None, t, X_d, Omp_d)
##completeEvent.wait()
##RES_h = cl_array.to_host(queue, RES_d)
##print RES_h


precode =  """
        #define complex_ctr(x, y) (float2)(x, y)
        #define complex_add(a, b) complex_ctr((a).x + (b).x, (a).y + (b).y)
        #define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
        #define complex_mul_scalar(a, b) complex_ctr((a).x * (b), (a).y * (b))
        #define complex_div_scalar(a, b) complex_ctr((a).x / (b), (a).y / (b))
        #define conj(a) complex_ctr((a).x, -(a).y)
        #define conj_transp(a) complex_ctr(-(a).y, (a).x)
        #define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))
        """
