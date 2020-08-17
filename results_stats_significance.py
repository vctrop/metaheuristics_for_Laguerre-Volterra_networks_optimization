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

# Third party
import numpy as np
import scipy.stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt


cmap = ['1', '#fb6a4a', 'mediumseagreen', 'limegreen', 'palegreen']
heatmap_args = {'cmap':cmap, 'linewidths': 0.5, 'linecolor': '0.3', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

# Verification of statistical tests with known distributions
# x = np.array([np.random.normal(loc=00.0, scale=1.0, size=30),
              # np.random.normal(loc=19.5, scale=1.0, size=30),
              # np.random.normal(loc=20.0, scale=1.0, size=30)])
# print(np.shape(x))
# fr_test = scipy.stats.friedmanchisquare(*x)
# print(fr_test)
# nm_posthoc = sp.posthoc_nemenyi_friedman(x.T)
# print(nm_posthoc)
# sp.sign_plot(nm_posthoc, **heatmap_args)
# exit()

function_evals = [i * 100 for i in range(1,101)] + [11000 + i * 1000 for i in range(90)]
function_evals_of_interest = [1e3, 1e4, 1e5]
evals_mask = [eval in function_evals_of_interest for eval in function_evals]

for system_order in ['finite', 'infinite']:
    print(system_order.upper())

    for eval in range(len(function_evals_of_interest)):
        algorithms_at_fes = []      
        for algorithm in ['sa', 'acfsa', 'pso', 'aiwpso', 'acor', 'baacor']:
            print(algorithm)
            # Load test costs of a given metaheuristic for a given system, considering some number of objective function evaluations
            base_filename   = './results/' + algorithm + '_' + system_order
            test_costs_mat  = np.load(base_filename + '_test_costs.npy')
            test_costs_of_interest = test_costs_mat[:, evals_mask]
            costs_fe = test_costs_of_interest[:, eval]
            
            algorithms_at_fes.append(list(costs_fe))
            print(str(function_evals_of_interest[eval]) + ':  \t' + str(np.mean(costs_fe)))
        
        algorithms_at_fes = np.array(algorithms_at_fes)
        print('\n Statistical significance')
        print(np.shape(algorithms_at_fes))    
        print('Friedman p-val = ' + str(scipy.stats.friedmanchisquare(*algorithms_at_fes)[1]) + '\n\n')
        nm_posthoc = sp.posthoc_nemenyi_friedman(algorithms_at_fes.T)
        plt.figure()
        sp.sign_plot(nm_posthoc, **heatmap_args)
        plt.show() 
        print('\n') 
           
           
