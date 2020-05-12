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
    offset = candidate_solution[(H * L + 1) + H * Q]
    
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
    # modified_variable indicates which parameters were modified in the solution. -1 if all of them were.
    def compute_cost(candidate_solution, modified_variable):
        
        # IO
        train_input, train_output = data_handling.read_io(train_filename)

        # Get parameters from candidate solution
        alpha, W, C, offset = decode_solution(candidate_solution, L, H, Q)
        
        # If the weights were modified, set flag so LVN normalizes weights and scales coefficients before output computation 
        if modified_variable == -1 or (modified_variable >= 1 and modified_variable <= L * H):
            weights_modified = True
        else:
            weights_modified = False
            
        # Generate output and compute cost
        solution_system = laguerre_volterra_network_structure.LVN()
        solution_system.define_structure(L, H, Q, 1/Fs)
        solution_output = solution_system.compute_output(train_input, alpha, W, C, offset, weights_modified)
        
        cost = NMSE(train_output, solution_output, alpha)
        
        return cost
        
    return compute_cost


    
