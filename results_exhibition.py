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
