__author__ = 'Joao Costa'

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la
from pyopencl.elementwise import ElementwiseKernel
import matplotlib.pyplot as plt

"""
Solve the problem
Xi' = M1*Xi + M2*Xi~
M1 and M2 are 6*6 Complex Matrixes
Xi in the form [P11i, P22i, P33i, P21i, P31i, P32i] where Pxyi is a complex number
""" 

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

# Constants 
M = 10 # Number of atoms
L = 1E-9 # Atom Spacing
N = 1000 # Number of time intervals
dt = np.float32(0.01) # Time interval
Timeline = np.arange(0.0, N, dt) 
Po = np.float32(1.0)
Delta = np.float32(1.0)
Gama = np.float32(1.0)
Omc = np.float32(1.0)

# acabar os defines dos valores
text = "#define M " 
f = open("constants.cl",'w+')
f.write(str(M))

#Initial Conditions OmegaP is yet missing here 
P11 = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P22 = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P33 = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P21 = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P31 = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P32 = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
Omp_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)

X_h = []
for i in range(M):
    X_h.append(np.array([P11[i], P22[i], P33[i], P21[i], P31[i], P32[i], 0.0, 0.0]))
X_h = np.array(X_h)

X_d = cl_array.to_device(queue,X_h)
Omp_d = cl_array.to_device(queue,Omp_h)
# print X_h

f = open("kernel.cl", "r")
Source = f.read()
prg = cl.Program(ctx, Source).build()

for t in Timeline:
    t = np.float32(t)
    completeEvent = prg.RK4Step(queue, (M,), None, t, X_d, Omp_d)
    completeEvent.wait()
