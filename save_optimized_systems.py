#!python3

# Copyright (C) 2020 Victor O. Costa

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

# Python std library
import sys
# 3rd party
import numpy as np
# Own
import optimization_utilities
import data_handling

# Argument number checking
if len(sys.argv) != 5:
    print('Error, wrong number of arguments. Execute this script as follows:\npython3 %s {simulated system order} {metaheuristic} {number of function evaluations (FE)} {run}' % sys.argv[0])
    print('The allowed values are: order = {\'finite\', \'infinite\'},  metaheuristic = {\'SA\', \'ACFSA\', \'PSO\', \'AIWPSO\', \'ACOr\', \'BAACOr\'}, run < 30, FE < 1e5')
    exit(-1)
    
# Argument coherence checking
order_str = (sys.argv[1]).lower()
if order_str != 'finite' and order_str != 'infinite':
    print('Error, choose either \'finite\' or \'infinite\' for the simulated system order')
    exit(-1)
    
metaheuristic_name = (sys.argv[2]).lower()
if metaheuristic_name != 'acor' and metaheuristic_name != 'baacor'  and metaheuristic_name != 'sa' and metaheuristic_name != 'acfsa' and metaheuristic_name != 'pso' and metaheuristic_name != 'aiwpso':
    print('Error, choose an available metaheuristic')
    exit(-1)
    
run = int(sys.argv[3])
if run < 0 or run > 29:
    print('Error, choose a run between 0 and 29')
    exit(-1)    

function_evals = [i * 100 for i in range(1,101)] + [11000 + i * 1000 for i in range(90)]
FE_of_interest = int(sys.argv[4])
if not (FE_of_interest in function_evals):
    print('Error, choose a valid number of function evaluations')
    exit(-1)
    

    
# Load the train solutions matrix (30, 190, 30)
train_solutions = np.load('./results/' + metaheuristic_name + '_' + order_str + '_train_solutions.npy')
# print(np.shape(train_solutions))

# Save solution of a run of some metaheuristic in a given system, in a certain point of the search
L = 5; H = 3; Q = 4
file_name = f'./signals_and_systems/optimized_sys_{order_str}_{metaheuristic_name}_run{run}_{FE_of_interest}FE'
evals_mask = [eval == FE_of_interest for eval in function_evals]
flat_solution = train_solutions[run, evals_mask, :][0]
print(flat_solution)
system_parameters = optimization_utilities.decode_solution(flat_solution, L, H, Q)

data_handling.write_LVN_file(file_name, system_parameters)





