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

# finite_acor_1k_solutions
# finite_acor_1k_test_costs
# finite_acor_1k_times

for system_order in ["finite", "infinite"]:
    print(system_order)
    for algorithm in ["sa", "pso", "acor"]:
        print(algorithm)
        for func_evals in ["1k", "5k", "10k", "50k", "100k"]:
            print(func_evals)
            base_filename = "./results/" + system_order + "_" + algorithm + "_" + func_evals + "_"
            
            # Load data
            train_times     = np.load(base_filename + "times.npy")
            found_solutions = np.load(base_filename + "solutions.npy")
            test_costs      = np.load(base_filename + "test_costs.npy")
            
            # Get train costs
            train_costs = []
            for solution in found_solutions:
                train_costs.append(solution[-1])
        
            # Compute avgs and sample stds
            train_times_avg = np.mean(train_times)
            train_costs_avg = np.mean(train_costs)
            test_costs_avg  = np.mean(test_costs)
            
            train_times_std = np.std(train_times, ddof=1)
            train_costs_std = np.std(train_costs, ddof=1)
            test_costs_std  = np.std(test_costs , ddof=1)
            
            print("Train time: %.3f (%.3f)" % (train_times_avg, train_times_std))
            print("Train cost: %.3f (%.3f)" % (train_costs_avg, train_costs_std))
            print("Test cost : %.3f (%.3f)" % (test_costs_avg , test_costs_std))