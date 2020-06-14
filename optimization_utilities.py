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
# Own
import data_handling
import laguerre_volterra_network_structure
# Third party
import numpy as np

# Normalized mean squared error
def NMSE(y, y_pred, alpha):
    if len(y) != len(y_pred):
        print('Actual and predicted y have different lengths')
        exit(-1)
    
    # Laguerre alpha paremeter determines the system memory (find reference for formula)
    M = laguerre_volterra_network_structure.laguerre_filter_memory(alpha)
    
    if len(y) <= M:
        print('Data length is less than required by alpha parameter')
        exit(-1)
    
    y = np.array(y)
    y_pred = np.array(y_pred)
    error = y[M:] - y_pred[M:]
    
    NMSE = sum( error**2 ) / sum( y[M:]**2 )
    
    return NMSE
    
    
# Normalized mean squared error that does not consider alpha, used in dynamic cost function
def raw_NMSE(y, y_pred):
    if len(y) != len(y_pred):
        print('Error, actual and predicted y have different lengths')
        exit(-1)
        
    y = np.array(y)
    y_pred = np.array(y_pred)
    error = y - y_pred
    
    NMSE = sum( error**2 ) / sum( y**2 )
    
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
    
    
# Break flat list-like solution into [W, C, offset] for a given LVN structure
# No alpha in this solution encoding
def decode_alphaless_solution(candidate_solution, L, H, Q):
    # Identify solution members
    flat_W = candidate_solution[0 : (H * L)]
    flat_C = candidate_solution[(H * L) : (H * L) + H * Q]
    offset = candidate_solution[(H * L) + H * Q]
    
    # de-flat W and C
    W = []
    C = []
    for hidden_unit in range(H):
        W.append( flat_W[hidden_unit * L : (hidden_unit + 1) * L] )
        C.append( flat_C[hidden_unit * Q : (hidden_unit + 1) * Q] )
    
    return W, C, offset
    
    
# Define cost for a given LVN structure 
def define_cost(L, H, Q, Fs, train_filename):
    # IO
    train_input, train_output = data_handling.read_io(train_filename)
    #
    solution_system = laguerre_volterra_network_structure.LVN()
    solution_system.define_structure(L, H, Q, 1/Fs)
    # Compute cost of a candidate solution, which is encoded as a flat array: alpha, W(0,0) ... W(L-1,H-1), C(0,0) ... C(Q-1,H-1), offset
    # modified_variable indicates which parameters were modified in the solution. -1 if all of them were.
    def compute_static_NMSE(candidate_solution, modified_variable):
        # Get parameters from candidate solution
        alpha, W, C, offset = decode_solution(candidate_solution, L, H, Q)
        
        # If the weights were modified, set flag so LVN normalizes weights and scales coefficients before output computation 
        if modified_variable == -1 or (modified_variable >= 1 and modified_variable <= L * H):
            weights_modified = True
        else:
            weights_modified = False
            
        # Generate output and compute cost
        solution_output = solution_system.propagate_LVN(train_input, alpha, W, C, offset, weights_modified)
        
        cost = NMSE(train_output, solution_output, alpha)
        
        return cost
        
    return compute_static_NMSE


# Define dynamic cost for a given LVN structure 
# To determine how to split the train signal in windows, a hyperparameter signal_window_len_ratio defines how many times the signal length is greater than the windows length
def define_dynamic_cost(signal_window_len_ratio, alpha, L, H, Q, Fs, train_filename):
    # IO
    train_input, train_output = data_handling.read_io(train_filename)
    # Define windows length
    window_length = int(len(train_input) / signal_window_len_ratio)
    # Compute number of windows considering an overlap of 50% between successive windows
    num_windows = 2 * (len(train_input) / window_length) - 1
    
    # Propagate Laguerre filterbank
    solution_system = laguerre_volterra_network_structure.LVN()
    solution_system.define_structure(L, H, Q, 1/Fs)
    filterbank_outputs = solution_system.propagate_laguerre_filterbank(train_input, alpha)
    
    # Compute dynamic cost of a candidate solution, which is encoded as a flat array: alpha, W(0,0) ... W(L-1,H-1), C(0,0) ... C(Q-1,H-1), offset
    # This cost changes the portion of train signal under optimization according to the proportion of iterations already performed by the metaheuristic
    def compute_dynamic_NMSE(candidate_solution, iterations_proportion, modified_variable):
        # Get parameters from candidate solution
        W, C, offset = decode_alphaless_solution(candidate_solution, L, H, Q)
        
        # If the weights were modified, set flag so LVN normalizes weights and scales coefficients before output computation 
        if modified_variable == -1 or (modified_variable >= 1 and modified_variable <= L * H):
            weights_modified = True
        else:
            weights_modified = False
        
        # Determine current window
        current_window = int(math.floor( num_windows * iterations_proportion ))
        # Array positions defining the window
        win_min_pos = current_window * int(window_length / 2 )
        win_max_pos = win_min_pos + window_length
        
        if win_max_pos > len(train_input):
            print('Error, window index is larger than signal size')
            exit(-1)
        
        # Generate output for the current window under optimization and compute cost
        solution_output = solution_system.polynomial_activation(train_input[win_min_pos : win_max_pos], filterbank_outputs[win_min_pos : win_max_pos], W, C, offset, weights_modified)
        cost = raw_NMSE(train_output[win_min_pos : win_max_pos], solution_output)
        print(cost)
        
        return cost
        
    return compute_dynamic_NMSE