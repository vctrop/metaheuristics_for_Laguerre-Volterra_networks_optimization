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
import sys
# Third party modules
import numpy as np
import matplotlib.pyplot as plt
# Utilities
import optimization_utilities
import data_handling
# Metaheuristics
import ant_colony_for_continuous_domains
import simulated_annealing
import particle_swarm_optimization


if len(sys.argv) != 4:
    print("Error, enter the structure of the LVN to be optimized as follows:\n%s {L} {H} {Q}")
    exit(-1)
    
# Structural parameters  
Fs = 25  
L = int(sys.argv[1]);   H = int(sys.argv[2]);    Q = int(sys.argv[3]);

# Parameters to be optimized
alpha_min   = 1e-5; alpha_max   = 0.9   # estimated lag with alpha = 0.9 is 263
weight_min  = -1;   weight_max  = 1
coef_min    = -1;   coef_max    = 1  
offset_min  = -1;   offset_max  = 1
    
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
    
# Optimization
function_evals = [100.00]

# ACOr
# Total # of function evaluations: archive_size + population_size * num_iterations
print("ACOr")
m = 5; k = 50; q = 0.01; xi = 0.85
ACOr = ant_colony_for_continuous_domains.ACOr()
ACOr.set_verbosity(False)
ACOr.set_cost(optimization_utilities.define_cost(L, H, Q, Fs, "./signals_and_systems/infinite_order_train.csv"))
ACOr.set_parameters(m, k, q, xi, function_evals)
ACOr.define_variables(initial_ranges, is_bounded)
best_solution = ACOr.optimize()
print(best_solution)

# SA
# Total # of function evaluations: global_iter * local_iter + 1
print("SA")
SA_local_iterations = 50
T0 = 100.0; decay_constant = 0.99; step_size = 1e-2
SA = simulated_annealing.SA()
SA.set_verbosity(False)
SA.set_cost(optimization_utilities.define_cost(L, H, Q, Fs, "./signals_and_systems/infinite_order_train.csv"))
SA.set_parameters(T0, decay_constant, step_size,SA_local_iterations,function_evals)
SA.define_variables(initial_ranges, is_bounded)
best_solution = SA.optimize()
print(best_solution)

# PSO
# Total # of function evaluations: population_size * num_iterations
print("PSO")
swarm_size = 20; accel_p = 2; accel_g = 2
PSO = particle_swarm_optimization.PSO()
PSO.set_verbosity(False)
PSO.set_cost(optimization_utilities.define_cost(L, H, Q, Fs, "./signals_and_systems/infinite_order_train.csv"))
PSO.set_parameters(swarm_size, accel_p, accel_g, function_evals)
PSO.define_variables(initial_ranges, is_bounded)
best_solution = PSO.optimize()
print(best_solution)

# system_parameters = optimization_utilities.decode_solution(best_solution, L, H, Q)
# data_handling.write_LVN_file("pso_infinite", system_parameters)