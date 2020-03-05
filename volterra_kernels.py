#!python3

# MIT License
# Copyright (c) 2020 Victor O. Costa
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Given a Laguerre-Volterra Network, plot corresponding low-order Volterra kernels
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import laguerre_volterra_network_structure
import data_handling

if len(sys.argv) != 2:
    print("Error, specify the LVN file to extract kernels from")
    print("E.g. \"py volterra_kernels.py my_sys.LVN\" ")
    exit(-1)

def laguerre_convolution_kernel(order, sample, alpha):
    sum = 0.0
    for k in range(order):
        sum += ((-1) ** k) * binom(sample, k) * binom(order, k) * (alpha ** (order - k)) * ((1 - alpha) ** k)
    
    output = (alpha ** ((sample - order)/2)) * np.sqrt(1 - alpha) * sum
    return output
    
#LVN_parameters = data_handling.read_LVN_file("finite_ord_train_system.LVN")
LVN_parameters = data_handling.read_LVN_file(sys.argv[1])
print(np.shape(LVN_parameters[2]))
alpha = LVN_parameters[0]
W = np.array(LVN_parameters[1])
C = np.array(LVN_parameters[2])
offset = LVN_parameters[3]

L = len(W[0])
Q = len(C[0])
H = len(C)

memory = laguerre_volterra_network_structure.laguerre_filter_memory(alpha)

# Compute 1st order kernel
kernel_1 = np.array([0.0]*memory)
# At each kenel point
for m in range(memory):
    sum_units = 0.0
    # For each hidden unit
    for h in range(H):
        sum_filters = 0.0
        # For each laguerre filter
        for j in range(L):
            sum_filters += W[h][j] * laguerre_convolution_kernel(j, m, alpha)
            
        sum_units += C[h, 0] * sum_filters
    kernel_1[m] = np.float(sum_units)

print("[1st order Volterra kernel]")    
print(kernel_1)
plt.figure()
plt.plot(kernel_1)

# Compute 2nd order kernel
kernel_2 = np.array([[0.0] * memory for _ in range(memory)])
# At each kenel point
for m1 in range(memory):
    for m2 in range(memory):
        sum_units = 0.0
        # For each hidden unit
        for h in range(H):
            sum_filters = 0.0
            # For each laguerre filter
            for j1 in range(L):
                for j2 in range(L):
                    sum_filters += W[h][j1] * W[h][j2] * laguerre_convolution_kernel(j1, m1, alpha) * laguerre_convolution_kernel(j2, m2, alpha)
            
            sum_units += C[h, 1] * sum_filters
        
        kernel_2[m1, m2] = sum_units
        
print("[2nd order Volterra kernel]")    
print(kernel_2)

fig = plt.figure()
ax = fig.gca(projection='3d')
x = y = np.arange(0, memory, 1)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, kernel_2, cmap='viridis', linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()