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
import math
import matplotlib.pyplot as plt

import laguerre_volterra_network_structure
import ant_colony_for_continuous_domains
import data_handling



# Normalized mean swquared error
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

# Compute cost of candidate solution
def compute_cost(candidate_solution):
    # IO
    train_input, train_output = data_handling.read_io("finite_ord_train.csv")
    
    # Structural parameters  
    Fs = 25  
    L = 5;   H = 1;    Q = 4;
    
    # Fazer conversão uni -> bidimensional aqui
    
    # Continuous parameters
    alpha = candidate_solution[0]
    w = [candidate_solution[1 : L+1]]
    c = [candidate_solution[L+1 : L+Q+1]]
    offset = candidate_solution[-1]
    
    # Generate output and compute cost
    solution_system = laguerre_volterra_network_structure.LVN()
    solution_system.define_structure(L, H, Q, 1/Fs)
    solution_output = solution_system.compute_output(train_input, alpha, w, c, offset, True)
    
    cost = NMSE(train_output, solution_output, alpha)
    
    return cost
    
# Parameters to be optimized
alpha_min   = 0;    alpha_max   = 0.9   # approx lag with 0.9 is 263
weight_min  = -1;  weight_max  = 1
coef_min    = -1;  coef_max    = 1  
offset_min  = -1;  offset_max  = 1
    
# Setup ACOr and optimize
colony = ant_colony_for_continuous_domains.ACOr()
num_iterations = 50
# Solution organization
# alpha, w0, ..., wL-1, c1, ..., cQ, offset

# TODO: PARAMETERIZE
ranges = [[alpha_min,alpha_max],
          [weight_min,weight_max],
          [weight_min,weight_max],
          [weight_min,weight_max],
          [weight_min,weight_max],
          [weight_min,weight_max],
          [coef_min,coef_max],
          [coef_min,coef_max],
          [coef_min,coef_max],
          [coef_min,coef_max],
          [offset_min,offset_max]]

colony.set_cost(compute_cost)
colony.set_parameters(num_iterations, 5, 50, 0.01, 0.85)
colony.set_variables(11, ranges)
solution = colony.optimize()
print(solution)
