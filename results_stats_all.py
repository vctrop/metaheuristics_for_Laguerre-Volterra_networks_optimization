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
            
            print("Train time: %g (%g)" % (train_times_avg, train_times_std))
            print("Train cost: %g (%g)" % (train_costs_avg, train_costs_std))
            print("Test cost : %g (%g)" % (test_costs_avg , test_costs_std))