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

zero = np.complex64(0.0)

X_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
Y_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)

DATA_h = []
for i in range(M):
    DATA_h.append(np.array([X[i], Y[i]]))
DATA_h = np.array(DATA_h)

# X_d = cl_array.to_device(queue, X_h)
# Y_d = cl_array.to_device(queue, Y_h)
# RES_d = cl_array.to_device(queue, np.empty_like(X_h))

DATA_d = cl_array.to_device(queue, DATA_h)
RES_d = cl_array.to_device(queue, np.empty_like(X_h))

precode =  """
        #pragma OPENCL EXTENSION cl_amd_printf : enable
        #define complex_ctr(x, y) (float2)(x, y)
        #define complex_add(a, b) complex_ctr((a).x + (b).x, (a).y + (b).y)
        #define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
        #define complex_mul_scalar(a, b) complex_ctr((a).x * (b), (a).y * (b))
        #define complex_div_scalar(a, b) complex_ctr((a).x / (b), (a).y / (b))
        #define conj(a) complex_ctr((a).x, -(a).y)
        #define conj_transp(a) complex_ctr(-(a).y, (a).x)
        #define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))
        """
        
df = ElementwiseKernel(ctx,
        "float2 *X, "
        "float2 *Y, "
        "float2 *res ",
        """
        res[i].x = X[i].x;
        res[i].y = -X[i].y
        """,
        "df",
        preamble=precode)

test = ElementwiseKernel(ctx,
        "float4 *data, "
        "float2 *res ",
        """
        res[i] = complex_add( data[i].s0, data[i].s1 )
        """,
        "test",
        preamble=precode)

# df(X_d, Y_d, RES_d)
# print "Numpy Result"
# print X_h
# print "PyOpenCL Result"
# print RES_d
test(DATA_d, RES_d)
# print "Numpy Result"
# print X_h + Y_h
# print "PyOpenCL Result"
# print RES_d