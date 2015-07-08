import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

M = 3

zero = np.complex64(0.0)

X1_h = np.array([1 + 1j*2, 2 + 1j*3, 3 + 1j*4]).astype(np.complex64)
X2_h = np.array([1 + 1j*2, 2 + 1j*3, 3 + 1j*4]).astype(np.complex64)
X3_h = np.array([1 + 1j*2, 2 + 1j*3, 3 + 1j*4]).astype(np.complex64)
Y1_h = np.array([4 + 1j*5, 5 + 1j*6, 6 + 1j*7]).astype(np.complex64)
Y2_h = np.array([4 + 1j*5, 5 + 1j*6, 6 + 1j*7]).astype(np.complex64)
Y3_h = np.array([4 + 1j*5, 5 + 1j*6, 6 + 1j*7]).astype(np.complex64)
aux_h = np.complex64(1 + 1j*1)
RES_h = np.empty_like(X1_h)
id_h = np.empty_like(X1_h)

dados_h = []
for i in range(3):
      dados_h.append(np.array([X1_h[i], X2_h[i], X3_h[i], Y1_h[i], Y2_h[i], Y3_h[i]]).astype(np.complex64))
dados_h = np.array(dados_h).astype(np.complex64)

print dados_h

aux_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=aux_h)
dados_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=dados_h)
RES_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf = RES_h)
id_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf = id_h)


Source = """
__kernel void soma( __global float8 *dados, __global float2 *id, __global float2 *res){
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);
	id[gid_x].x = gid_x;
	id[gid_x].y = gid_y;
	res[gid_x] = dados[gid_x];
}
"""
prg = cl.Program(ctx, Source).build()

completeEvent = prg.soma(queue, (M,), None, dados_d, id_d, RES_d)
completeEvent.wait()

cl.enqueue_copy(queue, RES_h, RES_d)
cl.enqueue_copy(queue, id_h, id_d)
print "GPU ID"
print id_h
print "GPU RES"
print RES_h
