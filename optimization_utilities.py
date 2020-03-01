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
import data_handling
import laguerre_volterra_network_structure

# Normalized mean squared error
def NMSE(y, y_pred, alpha):
    if len(y) != len(y_pred):
        print("Actual and predicted y have different lengths")
        exit(-1)
    
    # Laguerre alpha paremeter determines the system memory (find reference for formula)
    M = laguerre_volterra_network_structure.laguerre_filter_memory(alpha)
    
    if len(y) <= M:
        print("Data length is lesser than required by alpha parameter")
        exit(-1)
    
    y = np.array(y)
    y_pred = np.array(y_pred)
    error = y[M:] - y_pred[M:]
    
    NMSE = sum( error**2 ) / sum( y[M:]**2 )
    
    return NMSE

    
# Break flat list-like solution into [alpha, W, C, offset] for a given LVN structure
def decode_solution(candidate_solution, L, H, Q):
    # Identify solution members
    alpha = candidate_solution[0]
    flat_W = candidate_solution[1 : (H * L + 1)]
    flat_C = candidate_solution[(H * L + 1) : (H * L + 1) + H * Q]
    offset = candidate_solution[-1]
    
    # de-flat W and C
    W = []
    C = []
    for hidden_unit in range(H):
        W.append( flat_W[hidden_unit * L : (hidden_unit + 1) * L] )
        C.append( flat_C[hidden_unit * Q : (hidden_unit + 1) * Q] )
    
    return alpha, W, C, offset
    
# Compute cost of candidate solution, which is encoded as a flat array: alpha, W(0,0) ... W(L-1,H-1), C(0,0) ... C(Q-1,H-1), offset
def define_cost(L, H, Q, Fs, train_filename):
    # Cost computation parameterized by the nesting function (define_cost)
    def compute_cost(candidate_solution, weights_modified):
        # IO
        train_input, train_output = data_handling.read_io(train_filename)

        # Get parameters from candidate solution
        alpha, W, C, offset = decode_solution(candidate_solution, L, H, Q)
        
        # Generate output and compute cost
        solution_system = laguerre_volterra_network_structure.LVN()
        solution_system.define_structure(L, H, Q, 1/Fs)
        solution_output = solution_system.compute_output(train_input, alpha, W, C, offset, weights_modified)
        
        cost = NMSE(train_output, solution_output, alpha)
        
        return cost
        
    return compute_cost


    
