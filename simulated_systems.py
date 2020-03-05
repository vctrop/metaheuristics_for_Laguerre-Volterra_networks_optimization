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

import numpy as np
from laguerre_volterra_network_structure import LVN

# Sampling frequency fixed at 25 Hz
Fs = 25

# Simulate Laguerre-Volterra Network of arbitrary structure with randomized parameters
def simulate_LVN_random(input_data, L, H, Q):
    # Continuous parameters
    alpha = np.random.uniform(0, 0.5)  
    W = [list(np.random.random(L) * 2 - 1) for _ in range(H)]
    C = [list(np.random.random(Q) * 4 - 2) for _ in range(H)]
    offset = np.random.random()
    system_parameters = [alpha, W, C, offset]
    
    system = LVN()
    system.define_structure(L, H, Q, 1/Fs)
    output_data = system.compute_output(input_data, alpha, W, C, offset, False)
    
    return output_data, system_parameters
    
    
def simulate_LVN_deterministic(input_data, L, H, Q, parameters):
    alpha = parameters[0]
    W = parameters[1]
    C = parameters[2]
    offset = parameters[3]
    
    system = LVN()
    system.define_structure(L, H, Q, 1/Fs)
    output_data = system.compute_output(input_data, alpha, W, C, offset, False)
    
    return output_data


def simulate_LVN_geng(input_data):
    alpha = 0.1
    W = [[1, 0, 1, 1], [-1, 1, 2, 0.5]]
    C = [[1, -1], [1, 0.5]]
    offset = 0.0
    
    parameters = [alpha, W, C , offset]
    
    L = len(W[0])
    H = len(W)
    Q = len(C[0])
    
    output = simulate_LVN_deterministic(input_data, L, H, Q, parameters)
    
    return output, parameters