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
import sys 
import pickle as pkl
import time
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
if len(sys.argv) != 5:
    print("Error, wrong number of arguments. Execute this script as follows:\npython %s {simulated system order} {# of function evaluations} {metaheuristic} {output base filename}" % sys.argv[0])
    print("The allowed values are: order = {\"finite\", \"infinite\"}, 0 < F.E. < 100k, metaheuristic = {\"ACOr\", \"SA\", \"ACFSA\", \"PSO\", \"AIWPSO\"}")
    exit(-1)
    
# Argument coherence checking
if sys.argv[1] != "finite" and sys.argv[1] != "infinite":
    print("Error, choose either \"finite\" or \"infinite\" for the simulated system order")
    exit(-1)
    
# if (int(sys.argv[2]) % 500) != 0:
    # print("Error, number of objective function evaluations must be a multiple of 500")
    # exit(-1)
    
metaheuristic_name = (sys.argv[3]).lower()

if metaheuristic_name != "acor" and metaheuristic_name != "sa" and metaheuristic_name != "acfsa" and metaheuristic_name != "pso" and metaheuristic_name != "aiwpso":
    print("Error, choose an available metaheuristic")
    exit(-1)
    
output_base_filename = sys.argv[4]

# Filenames for train and test signals
train_filename = None
test_filename = None

# Optimized LVN structure definition
Fs = 25                             # Sampling frequency is assumed to be 25 Hz, but could be any other value
L = None;   H = None;    Q = None;    

# Whether the simulated system has finite or infinite order determines the structure of the optimized LVN and from which file the data will be loaded
if sys.argv[1] == "finite":
    train_filename = "finite_order_train.csv"
    test_filename = "finite_order_test.csv"
    L = 5;  H = 3;  Q = 4; 
else:
    # Inifinite order system not defined yet
    exit(-1)

# Number of objective function evaluations for this run    
num_func_evals = int(sys.argv[2])

# Instantiate metaheuristic
# In the order of hundreds of thousands of objective function evaluations, the evaluations for initialization of ACOr and SA are insignificant and not considered in the counting    
metaheuristic = None
if metaheuristic_name == "acor":
    print("ACOr")
    # Parameters used for ACOr
    k = 50;  pop_size = 5;  q = 0.01; xi = 0.85
    # Number of function evaluations for ACOr: pop_size * num_iterations
    num_iterations = num_func_evals / pop_size
    print("# iterations = %d" % num_iterations) 
    if not (num_iterations.is_integer()):
        print("Error, number of function evaluations is not divisible by population size")
        exit(-1)
    metaheuristic = ant_colony_for_continuous_domains.ACOr()
    metaheuristic.set_parameters(int(num_iterations), pop_size, k, q, xi)

elif metaheuristic_name == "sa":
    print("SA")
    # Parameters to be used for SA
    initial_temperature = 100.0;  cooling_constant = 0.99;  step_size = 1e-2;
    # Number of function evaluations for SA: global_iterations * local_iterations
    local_iterations = 500
    global_iterations = num_func_evals / local_iterations
    print("# local/global iterations = %d/%d" % (local_iterations, global_iterations)) 
    if not (global_iterations.is_integer()):
        print("Error, number of function evaluations is not divisible by number of local iterations")
        exit(-1)
    metaheuristic = simulated_annealing.SA()
    metaheuristic.set_parameters(int(global_iterations), int(local_iterations), initial_temperature, cooling_constant, step_size)
    
elif metaheuristic_name == "acfsa":
    print("ACFSA")
    # Parameters to be used for ACFSA
    initial_temperature = 100.0;  cooling_constant = 0.99
    # Number of function evaluations for ACFSA: global_iterations * local_iterations
    local_iterations = 500
    global_iterations = num_func_evals / local_iterations
    print("# local/global iterations = %d/%d" % (local_iterations, global_iterations)) 
    if not (global_iterations.is_integer()):
        print("Error, number of function evaluations is not divisible by number of local iterations")
        exit(-1)
    metaheuristic = simulated_annealing.ACFSA()
    metaheuristic.set_parameters(int(global_iterations), int(local_iterations), initial_temperature, cooling_constant)
    
elif metaheuristic_name == "pso":
    print("PSO")
    # Parameters to be used for PSO
    swarm_size = 20;  personal_acceleration = 2;  global_acceleration = 2
    # Number of function evaluations for PSO: swarm_size * num_iterations
    num_iterations = num_func_evals / swarm_size
    print("# iterations = %d" % num_iterations) 
    if not (num_iterations.is_integer()):
        print("Error, number of function evaluations is not divisible by swarm size")
        exit(-1)
    metaheuristic = particle_swarm_optimization.PSO()
    metaheuristic.set_parameters(int(num_iterations), swarm_size, personal_acceleration, global_acceleration)
    
else: # metaheuristic_name == "aiwpso"
    print("AIWPSO")
    # Parameters to be used for AIWPSO
    swarm_size = 20;  personal_acceleration = 2;  global_acceleration = 2; min_inertia = 0; max_inertia = 1
    # Number of function evaluations for PSO: swarm_size * num_iterations
    num_iterations = num_func_evals / swarm_size
    if not (num_iterations.is_integer()):
        print("Error, number of function evaluations is not divisible by swarm size")
        exit(-1)
    metaheuristic = particle_swarm_optimization.AIWPSO()
    metaheuristic.set_parameters(int(num_iterations), swarm_size, personal_acceleration, global_acceleration, min_inertia, max_inertia)
    
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
#  to a file with name defined in the last argument of the script
found_solutions = []
# For each found solution, compute cost function on test set
test_input, test_output = data_handling.read_io(test_filename)
test_costs = []
# Keep how much seconds each call to .optimize() spends
optimization_times = []
for i in range(30):
    # Search parameters on train set
    print("Round %d" % i)
    time_start = time.perf_counter()
    solution = metaheuristic.optimize()
    time_end = time.perf_counter()
    # Keep time spent
    optimization_times.append(time_end - time_start)
    # Keep solution found
    found_solutions.append(solution)
    
    # Decode solution and evaluate parameters on test set
    alpha, W, C, offset = optimization_utilities.decode_solution(solution, L, H, Q)
    LVN = laguerre_volterra_network_structure.LVN()
    LVN.define_structure(L, H, Q, 1/Fs)
    test_out_prediction = LVN.compute_output(test_input, alpha, W, C, offset, True)
    test_nmse = optimization_utilities.NMSE(test_output, test_out_prediction, alpha)
    test_costs.append(test_nmse) 

pkl.dump(optimization_times,    open("./results/" + output_base_filename + "_times.pkl", "wb"))    
pkl.dump(found_solutions,       open("./results/" + output_base_filename + "_solutions.pkl","wb"))
pkl.dump(test_costs,            open("./results/" + output_base_filename + "_test_costs.pkl","wb"))