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
# 3rd party
import numpy as np
import matplotlib.pyplot as plt

function_evals = [i * 100 for i in range(1,101)] + [11000 + i * 1000 for i in range(90)]
metaheuristics_names = ['sa', 'acfsa', 'pso', 'aiwpso', 'acor', 'baacor']

for system_order in ['finite', 'infinite']:
    plt.figure()    
    for index, metaheuristic_str in enumerate(metaheuristics_names):
        # Load (30,190) matrix with test costs
        base_filename   = './results/' + metaheuristic_str + '_' + system_order
        test_costs_matrix  = np.load(base_filename + '_test_costs.npy')
        # print(np.shape(test_costs_matrix))
        
        # Plot average cost history for the given metaheuristic
        average_cost_trajectory = np.sum(test_costs_matrix, axis=0)
        average_cost_trajectory /= 30
        plt.plot(function_evals, average_cost_trajectory, label=metaheuristic_str, linewidth=3)

    plt.xlabel('AFO', fontsize=18)
    plt.ylabel('Erro médio', fontsize=18)
    plt.legend(fontsize=16)

    plt.show()