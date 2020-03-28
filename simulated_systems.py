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

# Python standard library
import math
# Third party
import numpy as np
# LVN
from laguerre_volterra_network_structure import LVN

# Sampling frequency fixed at 25 Hz
Fs = 25

# Simulate Laguerre-Volterra Network of arbitrary structure with randomized parameters, and return both output signal and the parameters used (to use the same set of parameters in the test set)
def simulate_LVN_random(input_signal, L, H, Q):
    # Continuous parameters
    alpha = np.random.uniform(0, 0.5)  
    W = [list(np.random.random(L) * 2 - 1) for _ in range(H)]
    C = [list(np.random.random(Q) * 4 - 2) for _ in range(H)]
    offset = np.random.random()
    system_parameters = [alpha, W, C, offset]
    
    system = LVN()
    system.define_structure(L, H, Q, 1/Fs)
    output_signal = system.compute_output(input_signal, alpha, W, C, offset, False)
    
    return output_signal, system_parameters
    

# Simulate LVN of arbitrary structure with deterministic parameters, and return output signal
def simulate_LVN_deterministic(input_signal, L, H, Q, parameters):
    alpha = parameters[0]
    W = parameters[1]
    C = parameters[2]
    offset = parameters[3]
    
    system = LVN()
    system.define_structure(L, H, Q, 1/Fs)
    output_signal = system.compute_output(input_signal, alpha, W, C, offset, False)
    
    return output_signal

# Simulated infinite order (in Taylor and Volterra senses) system via a cascade of IIR filter and static exponential 
# Sum of exponentially weighted moving averages (EWMAs) as IIR  
def simulate_cascaded_random(input_signal, num_ewmas):
    
    # randomize alphas
    alphas = np.random.uniform(1e-5, 1, num_ewmas)
    #
    ewmas = [0 for _ in range(num_ewmas)]
    #
    output_signal = []
    for data_i in range(len(input_signal)):
        # 
        ewmas_sum = 0
        for ewma_i in range(num_ewmas):
            ewmas[ewma_i] = (1 - alphas[ewma_i]) * input_signal[data_i] + alphas[ewma_i] * ewmas[ewma_i]
            ewmas_sum += ewmas[ewma_i]
        
        y = math.exp(ewmas_sum) * math.sin(ewmas_sum)
        output_signal.append(y)
    
    
    return output_signal, alphas
    
#
def simulate_cascaded_deterministic(input_signal, alphas):
    num_ewmas = len(alphas)
    #
    ewmas = [0 for _ in range(num_ewmas)]
    ewmas_inspect = [[0 for _ in range(num_ewmas)] for _ in range(len(input_signal))]
    #
    output_signal = []
    for data_i in range(len(input_signal)):
        # 
        ewmas_sum = 0
        for ewma_i in range(num_ewmas):
            ewmas[ewma_i] = (1 - alphas[ewma_i]) * input_signal[data_i] + alphas[ewma_i] * ewmas[ewma_i]
            ewmas_sum += ewmas[ewma_i]
            ewmas_inspect[data_i][ewma_i] = ewmas[ewma_i]
        
        y = math.exp(ewmas_sum) * math.sin(ewmas_sum)
        output_signal.append(y)
    
    return output_signal
    