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
import matplotlib.pyplot as plt
import ant_colony_for_continuous_domains
import optimization_utilities
import data_handling

# Structural parameters  
Fs = 25  
L = 4;   H = 2;    Q = 2;

# Parameters to be optimized
alpha_min   = 1e-5; alpha_max   = 0.9   # estimated lag with alpha = 0.9 is 263
weight_min  = -1;   weight_max  = 1
coef_min    = -1;   coef_max    = 1  
offset_min  = -1;   offset_max  = 1
    
# Setup ACOr and optimize
num_iterations = 1000

# Define the ranges to be used in random initialization of algorithms for each variable,
#  along with which variables are bounded by these ranges during the optimization
initial_ranges = []
is_bounded = []

# Alpha variable is bounded
initial_ranges.append([alpha_min,alpha_max])
is_bounded.append(True)
# Hidden units input weights are forcedly bounded by l2-normalization (normalization to unit Euclidean norm)
for _ in range(L * H): 
    initial_ranges.append([weight_min, weight_max])
    is_bounded.append(False)
# Polynomial coefficients are not bounded in the initial range
for _ in range(Q * H):
    initial_ranges.append([coef_min,coef_max])
    is_bounded.append(False)
# Output offset is not bounded in the initial range
initial_ranges.append([offset_min, offset_max])
is_bounded.append(False)
    
colony = ant_colony_for_continuous_domains.ACOr()
colony.set_verbosity(True)
colony.set_cost(optimization_utilities.define_cost(L, H, Q, Fs, "geng_train.csv"))
colony.set_parameters(num_iterations, 5, 50, 0.01, 0.85)
colony.define_variables(initial_ranges, is_bounded)
solution = colony.optimize()

#system_parameters = optimization_utilities.decode_solution(solution, L, H, Q)
#data_handling.write_LVN_file("acor_1k_geng", system_parameters)