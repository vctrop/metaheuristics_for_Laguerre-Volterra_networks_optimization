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
from scipy.stats import wilcoxon

# finite_acor_1k_solutions
# finite_acor_1k_test_costs
# finite_acor_1k_times

for system_order in ["finite", "infinite"]:
    print(system_order)
    for func_evals in ["1k", "5k", "10k", "50k", "100k"]:
        print(func_evals)
        
        pso_base_filename =  "./results/" + system_order + "_" + "pso" + "_" + func_evals + "_"
        acor_base_filename = "./results/" + system_order + "_" + "acor" + "_" + func_evals + "_"
        sa_base_filename = "./results/" + system_order + "_" + "sa" + "_" + func_evals + "_"
        
        # Load data
        pso_test_costs      = np.load(pso_base_filename + "test_costs.npy")
        acor_test_costs      = np.load(acor_base_filename + "test_costs.npy")
        sa_test_costs      = np.load(sa_base_filename + "test_costs.npy")
        
        
        if system_order == "finite" and func_evals == "100k":
            stat, p = wilcoxon(acor_test_costs, sa_test_costs)
        else:
            stat, p = wilcoxon(acor_test_costs, pso_test_costs)
        
        print("p-value = %f" % (p))
        