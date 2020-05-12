#!python3

# Copyright (C) 2020  Victor O. Costa

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    alphas = np.random.uniform(0.2, 0.8, num_ewmas)
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
        
        y = math.exp(math.sin(ewmas_sum))
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
        
        y = math.exp(math.sin(ewmas_sum))
        output_signal.append(y)
    
    return output_signal
    