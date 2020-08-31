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

function_evals = [i * 100 for i in range(1,101)] + [11000 + i * 1000 for i in range(90)]
function_evals_str = [str(i) for i in function_evals]
function_evals_of_interest = [1e2, 1e3, 5e3,1e4,1e5]
evals_mask = [eval in function_evals_of_interest for eval in function_evals]

print('Function evaluations of interest:\n' + str(function_evals_of_interest))

for system_order in ['finite', 'infinite']:
    print(system_order)
    for algorithm in ['sa', 'acfsa', 'pso', 'aiwpso', 'acor', 'baacor']:
        print(algorithm)
        base_filename = './results/' + algorithm + '_' + system_order
        
        # Load data
        train_times     = np.load(base_filename + '_times.npy')
        train_costs_mat = np.load(base_filename + '_train_costs.npy')
        test_costs_mat  = np.load(base_filename + '_test_costs.npy')
        
        train_times_avg = np.mean(train_times)
        train_times_std = np.std(train_times, ddof=1)
        print('Train time: %.3f (%.3f)' % (train_times_avg, train_times_std))
        
        train_costs_at_fes = train_costs_mat[:, evals_mask]
        # print(np.shape(train_costs_at_fes))
        test_costs_at_fes = test_costs_mat[:, evals_mask]
        # print(np.shape(train_costs_at_fes))
        
        if np.shape(train_costs_at_fes) != np.shape(test_costs_at_fes):
            print('Error, train and test costs have different shapes')
            exit(-1)
        
        
        # Compute avgs and sample stds for each function evaluation of interest (i.e. each column of the matrices)
        for c in range(np.shape(train_costs_at_fes)[1]):
            #print(function_evals_str[c])
            train_costs = train_costs_at_fes[:, c]
            # print(np.shape(train_costs))
            test_costs = test_costs_at_fes[:, c]
            # print(np.shape(test_costs))
            
            train_costs_avg = np.mean(train_costs)
            test_costs_avg  = np.mean(test_costs)
            train_costs_std = np.std(train_costs, ddof=1)
            test_costs_std  = np.std(test_costs , ddof=1)
            
            print('Train cost: %.3f' % (train_costs_avg))
            print('Test cost : %.3f (%.3f)' % (test_costs_avg , test_costs_std))
            print('\n')
        print('\n')
        
        