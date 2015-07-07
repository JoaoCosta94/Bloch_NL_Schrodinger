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

X_h = np.array([1 + 1j*1, 2 + 1j*2, 3 + 1j*3]).astype(np.complex64)
Y_h = np.array([1 + 1j*1, 2 + 1j*2, 3 + 1j*3]).astype(np.complex64)

Source = """
#define complex_ctr(x, y) (float2)(x, y)
#define complex_add(a, b) complex_ctr((a).x + (b).x, (a).y + (b).y)
#define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
#define complex_mul_scalar(a, b) complex_ctr((a).x * (b), (a).y * (b))
#define complex_div_scalar(a, b) complex_ctr((a).x / (b), (a).y / (b))
#define conj(a) complex_ctr((a).x, -(a).y)
#define conj_transp(a) complex_ctr(-(a).y, (a).x)
#define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))
#define complex_unit (float2)(0, 1)

__kernel void soma(__global float4 *data), __global float2 *res){
	const int gid = get_global_id(0);
	res[gid].x = data[gid].s0 + data[gid].s2;  
	res[gid].x = data[gid].s1 + data[gid].s3;  
}
"""
prg = cl.Program(ctx, Source).build()

DATA_h = []
for i in range(M):
      DATA_h.append(np.array([X_h[i], Y_h[i]]))
#       DATA_h.append(np.array([X_h[i]]))
DATA_h = np.array(DATA_h)
DATA_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=DATA_h)
RES_h = np.empty_like(DATA_h)
RES_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf = RES_h)

completeEvent = prg.soma(queue, (4,), None, DATA_d, RES_d)
completeEvent.wait()

# for i in range(M):
#     RES_h[i] = DATA_h[i].real + DATA_h[i].imag
cl.enqueue_copy(queue, RES_h, RES_d)
print DATA_h
print "GPU"
print RES_h
