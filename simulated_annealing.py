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

class SA:
    """ Class for the Simulated Annealing optimizer, following (Kirkpatrick et al., 1983) and using exponential decay cooling schedule (Nourani and Andresen, 1998) """
        
    def __init__(self):
        self.verbosity = True
        
        # Initial algorithm parameters
        self.num_global_iter = 0                        # Maximum number of global iterations
        self.num_local_iter = 0                         # Maximum number of local iterations
        self.temperature = 100                          # Temperature, which determines acceptance probability of Metropolis sampling
        self.cooling_constant = 0.99                    # Parameter for the exponential decay cooling schedule
        self.step_size = 1e-2                           # Technically each variable type has its own step size, but (Geng and Marmarelis, 2016) uses the same step for all variables
        
        # Initial (NULL) problem definition
        self.num_var = None                             # Number of variables
        self.var_ranges = []                            # Variables boundaries
        self.cost_function = None                       # Cost function to guide the search
        
        # Optimization results
        self.current_parameters = None                  # Set of parameters that define the current best solution
        self.current_cost = None                        # Cost associated with the current best solution

    def set_verbosity(self, status):
        """ If status is True, will print partial results during the search """
        if not (type(status) is bool):
            print("Error, verbosity parameter must be a boolean")
            exit(-1)
        self.verbosity = status    
        
        
    def set_parameters(self, num_global_iter, num_local_iter, initial_temperature, cooling_constant, step_size):
        """ Define values for the parameters used by the algorithm """
        
        if num_global_iter <= 0 or num_local_iter <= 0:
            print("Number of global and local iterations must be greater than zero")
            exit(-1)
            
        self.num_global_iter = num_global_iter    
        self.num_local_iter = num_local_iter     
        self.temperature = initial_temperature      
        self.cooling_constant = cooling_constant
        self.step_size = step_size
        
        
    def set_variables(self, ranges):
        """ Sets the number of variables and their boundaries when constrained. Else, ranges are (-inf, +inf) """
        # Error checking
        if self.num_iter == 0:
            print("Error, please set algorithm parameters before variables definition")
            exit(-1)
        if len(ranges) == 0:
            print("Error, ranges must have length > 0")
            exit(-1)
        
        self.num_var = len(ranges)
        self.var_ranges = ranges
        self.current_parameters = np.zeros(self.num_var + 1)          # The current solution
        
    
    def set_cost(self, cost_function):
        """ Define the cost function that will guide the search procedure """
        self.cost_function = cost_function
        
        
    def optimize(self):
        """ Generate a random initial solution and enter the algorithm loop until the number of global iterations is reached """
        
        # Initialize random solution and compute its cost
        
        
        # for global iterations
            # update temperature
            # for local iterations
                # generate candidate parameter set and compute its cost
                # decide if solution will replace the current one based on the Metropolis sampling algorithm
                
            
        
        
    
    