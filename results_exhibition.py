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

import sys
import numpy as np

if len(sys.argv) != 2:
    print("Error, please enter the base filename of the desired results")
    exit(-1)
    
file_basename = sys.argv[1]

# Load data
train_times      = np.load("./results/" + file_basename + "_times.npy")
found_solutions = np.load("./results/" + file_basename + "_solutions.npy")
test_costs      = np.load("./results/" + file_basename + "_test_costs.npy")

# Print computing times of each metaheuristic.optimization() call of each round
print("OPTIMIZATION TIMES")
print(train_times)
# Print solutions found for each round
print("FOUND SOLUTIONS")
print(found_solutions)
# Print NMSE on test data for the solutions of each round
print("TEST COSTS")
print(test_costs)
