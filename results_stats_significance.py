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
        