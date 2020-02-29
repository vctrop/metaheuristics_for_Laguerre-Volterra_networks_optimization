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

# Simulate 5th order laguerre expansion output weighted, summed and cascaded with 4th order algebraic polynomial equation 
def simulate_LVN(input_data):
    # Structural parameters for the described system
    L = 5;   H = 1;    Q = 4;
    # Continuous parameters (randomly choosen)
    alpha = 0.1
    w = [[0.33, 0.72, -0.46, -0.29, -0.91]]
    c = [[-0.53, 0.9, -1.81, 1.34]] 
    offset = 0
    
    system = LVN()
    system.define_structure(L, H, Q, 1/Fs)
    output_data = system.compute_output(input_data, alpha, w, c, offset, False)
    
    return output_data