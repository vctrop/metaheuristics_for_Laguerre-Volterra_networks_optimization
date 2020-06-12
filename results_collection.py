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
import time
# Third party 
import numpy as np

# Utilities
import optimization_utilities
import data_handling
# LVN
import laguerre_volterra_network_structure
# Metaheuristics
import ant_colony_for_continuous_domains
import simulated_annealing
import particle_swarm_optimization

# Argument number checking
if len(sys.argv) != 3:
    print('Error, wrong number of arguments. Execute this script as follows:\npython3 %s {simulated system order} {metaheuristic}' % sys.argv[0])
    print('The allowed values are: order = {\'finite\', \'infinite\'},  metaheuristic = {\'ACOr\', \'SA\', \'PSO\'}')
    exit(-1)
    
# Argument coherence checking
order_str = sys.argv[1] 
if order_str != 'finite' and order_str != 'infinite':
    print('Error, choose either \'finite\' or \'infinite\' for the simulated system order')
    exit(-1)
    
metaheuristic_name = (sys.argv[2]).lower()

if metaheuristic_name != 'acor' and metaheuristic_name != 'baacor'  and metaheuristic_name != 'sa' and metaheuristic_name != 'acfsa' and metaheuristic_name != 'pso' and metaheuristic_name != 'aiwpso':
    print('Error, choose an available metaheuristic')
    exit(-1)

# Filenames for train and test signals
train_filename = None
test_filename = None

# Optimized LVN structure definition
Fs = 25                             # Sampling frequency is assumed to be 25 Hz, but could be any other value
L = None;   H = None;    Q = None;    

# Whether the simulated system has finite or infinite order determines the structure of the optimized LVN and from which file the data will be loaded
train_filename = './signals_and_systems/' + order_str + '_order_train.csv'
test_filename  = './signals_and_systems/' + order_str + '_order_test.csv'
if order_str == 'finite':
    L = 5;  H = 3;  Q = 4
else:
    L = 2;  H = 4;  Q = 5

# Number of objective function evaluations of interest
function_evals = [i * 100 for i in range(101)] + [11000 + i * 1000 for i in range(90)]

# Instantiate metaheuristic
metaheuristic = None
if metaheuristic_name == 'acor':
    print('ACOr')
    # Parameters used for ACOr
    k = 50;  pop_size = 10;  q = 0.01; xi = 0.85
    metaheuristic = ant_colony_for_continuous_domains.ACOr()
    metaheuristic.set_parameters(pop_size, k, q, xi, function_evals)
    
elif metaheuristic_name == 'baacor':
    print('BAACOr')
    # Parameters used for BAACOr
    k = 50
    m = 10
    q_min = 1e-2;    q_max = 1.0
    xi_min = 0.1;    xi_max = 0.93
    # Configure
    metaheuristic = ant_colony_for_continuous_domains.BAACOr()
    metaheuristic.set_verbosity(False)
    metaheuristic.set_parameters(m, k, q_min, q_max, xi_min, xi_max, 'exp', 'sig', function_evals)

elif metaheuristic_name == 'sa':
    print('SA')
    # Parameters to be used for SA
    initial_temperature = 10.0;  cooling_constant = 0.99;  step_size = 1e-2;
    local_iterations = 100
    metaheuristic = simulated_annealing.SA()
    metaheuristic.set_parameters(initial_temperature, cooling_constant, step_size, local_iterations, function_evals)
    
elif metaheuristic_name == 'acfsa':
    print('ACFSA')
    # Parameters to be used for ACFSA
    local_iterations = 100
    initial_temperature = 50
    cooling_constant = 0.99 
    # Configure
    metaheuristic = simulated_annealing.ACFSA()
    metaheuristic.set_verbosity(False)
    metaheuristic.set_parameters(initial_temperature, cooling_constant, local_iterations, function_evals)


elif metaheuristic_name == 'pso':
    print('PSO')
    # Parameters to be used for PSO
    swarm_size = 20;  personal_acceleration = 2;  global_acceleration = 2
    metaheuristic = particle_swarm_optimization.PSO()
    metaheuristic.set_parameters(swarm_size, personal_acceleration, global_acceleration, function_evals)
    
else: # metaheuristic_name == "aiwpso":
    print("AIWPSO")
    # Parameters to be used for AIWPSO
    swarm_size = 20;  personal_acceleration = 2;  global_acceleration = 2
    min_inertia = 0.3; max_inertia = 0.99
    metaheuristic = particle_swarm_optimization.AIWPSO()
    metaheuristic.set_parameters(swarm_size, personal_acceleration, global_acceleration, min_inertia, max_inertia, function_evals)

# Cost function definition based on structural parameters and ground truth
metaheuristic.set_cost(optimization_utilities.define_cost(L, H, Q, Fs, train_filename))

# Define characteristics of variables to be optimized
# Variables initial ranges
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
# Hidden units input weights are forcedly bounded by l2-normalization (normalization to unit Euclidean norm), not by the metaheuristics
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

metaheuristic.define_variables(initial_ranges, is_bounded)
metaheuristic.set_verbosity(False)

# Run the metaheuristic 30 times and save results for the best found solution of each run
# For each found solution, compute cost function on test set
test_input, test_output = data_handling.read_io(test_filename)
train_solutions = []
train_costs = []
test_costs = []
LVN = laguerre_volterra_network_structure.LVN()
LVN.define_structure(L, H, Q, 1/Fs)
# Keep how much seconds each call to .optimize() spends
optimization_times = []

for i in range(30):
    # Search parameters on train set
    print('Round %d' % i)
    time_start = time.process_time()
    solutions_at_FEs = metaheuristic.optimize()
    time_end = time.process_time()
    # Keep time spent
    optimization_times.append(time_end - time_start)
    # Keep full solutions
    train_solutions.append(np.array(solutions_at_FEs))
    # Keep costs separate
    train_costs.append(solutions_at_FEs[:, -1])
    # Decode solution and evaluate parameters on test set
    run_test_NMSEs = []
    for solution in solutions_at_FEs:
        alpha, W, C, offset = optimization_utilities.decode_solution(solution, L, H, Q)
        test_out_prediction = LVN.propagate_LVN(test_input, alpha, W, C, offset, True)
        test_nmse = optimization_utilities.NMSE(test_output, test_out_prediction, alpha)
        run_test_NMSEs.append(test_nmse) 
    test_costs.append(run_test_NMSEs)

output_base_filename = metaheuristic_name + '_' + order_str
np.save('./results/' + output_base_filename + '_times.npy'          , optimization_times)   
np.save('./results/' + output_base_filename + '_train_solutions.npy', train_solutions)
np.save('./results/' + output_base_filename + '_train_costs.npy', train_costs)
np.save('./results/' + output_base_filename + '_test_costs.npy' , test_costs)
