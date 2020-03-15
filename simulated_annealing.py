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
import math

class SA:
    """ Class for the Simulated Annealing optimizer (Kirkpatrick et al., 1983), with pertubation on continuous variable as in (Geng and Marmarelis, 2016) and using exponential decay cooling schedule (Nourani and Andresen, 1998) """
        
    def __init__(self):
        self.verbosity = True
        
        # Initial algorithm parameters
        self.num_global_iter = 0                        # Maximum number of global iterations
        self.num_local_iter = 0                         # Maximum number of local iterations
        self.temperature = 100.0                        # Temperature, which determines acceptance probability of Metropolis sampling
        self.cooling_constant = 0.99                    # Parameter for the exponential decay cooling schedule
        self.step_size = 1e-2                           # Technically each variable type has its own step size, but (Geng and Marmarelis, 2016) uses the same step for all variables
        
        # Initial (NULL) problem definition
        self.num_variables = None                       # Number of variables
        self.initial_ranges = []                        # Initialization boundaries for each variable
        self.is_bounded = []                            # Here, if a variable is constrained, it will be limited to its initialization boundaries for all the search
        self.cost_function = None                       # Cost function to guide the search
        
        # Optimization results
        self.current_solution = None                    # Set of variables that define the current solution, with its cost as the last element of the list
        self.best_solution = None                       # Best solution encountered in the search process    
        
        
    def set_verbosity(self, status):
        """ If status is True, will print partial results during the search """
        # Input error checking
        if not (type(status) is bool):
            print("Error, verbosity parameter must be a boolean")
            exit(-1)
            
        self.verbosity = status    
        
        
    def set_parameters(self, num_global_iter, num_local_iter, initial_temperature, cooling_constant, step_size):
        """ Define values for the parameters used by the algorithm """
        # Input error checking
        if num_global_iter <= 0 or num_local_iter <= 0:
            print("Number of global and local iterations must be greater than zero")
            exit(-1)
            
        self.num_global_iter = num_global_iter    
        self.num_local_iter = num_local_iter     
        self.temperature = initial_temperature      
        self.cooling_constant = cooling_constant
        self.step_size = step_size
        
        
    def define_variables(self, initial_ranges, is_bounded):
        """ Defines the number of variables, their initial values ranges and wether or not these ranges constrain the variable during the search """
        # Input error checking
        if self.num_global_iter == 0 or self.num_local_iter == 0:
            print("Error, please set algorithm parameters before variables definition")
            exit(-1)
        if len(initial_ranges) == 0 or len(is_bounded) == 0:
            print("Error, initial_ranges and is_bounded lists must not be empty")
            exit(-1)
        if len(initial_ranges) != len(is_bounded):
            print("Error, the number of variables for initial_ranges and is_bounded must be equal")
            exit(-1)
        
        self.num_variables = len(initial_ranges)
        self.initial_ranges = initial_ranges
        self.is_bounded = is_bounded
        self.current_solution = np.zeros(self.num_variables + 1)
        
    
    def set_cost(self, cost_function):
        """ Define the cost function that will guide the search procedure """
        self.cost_function = cost_function
        
        
    def optimize(self):
        """ Generate a random initial solution and enter the algorithm loop until the number of global iterations is reached """
        # Input error checking
        if self.num_variables == None:
            print("Error, first set the number of variables and their boundaries")
            exit(-1)
        if self.cost_function == None:
            print("Error, first define the cost function to be used")
            exit(-1)
        
        # Randomize initial solution
        for i in range(self.num_variables):
            self.current_solution[i] = np.random.uniform(self.initial_ranges[i][0], self.initial_ranges[i][1])
        # Compute its cost considering that weights were modified
        self.current_solution[-1] = self.cost_function(self.current_solution, -1)
        self.best_solution = np.array(self.current_solution)

        if self.verbosity: print("[ALGORITHM MAIN LOOP]")
        # SA main loop
        for global_i in range(self.num_global_iter):
            # Update temperature according to the exponential decay cooling scheduling
            self.temperature = self.temperature * self.cooling_constant
            for local_i in range(self.num_local_iter):
                total_i = local_i + self.num_local_iter * global_i
                if self.verbosity:
                    print("[%d]" % total_i)
                    print(self.current_solution)
                
                ## Generate candidate solution and compute its cost
                # Random sign of pertubation
                random_sign = (-1) ** np.random.randint(0,2)
                # Choose which variable will be pertubated
                chosen_variable = np.random.randint(0, self.num_variables)      # [0, num_variables)
                # Pertubate the chosen variable according to the random sign and step size
                pertubated_variable = self.current_solution[chosen_variable] + random_sign * self.step_size
                
                # For bounded variables, deal with search space violation using the hard border strategy
                if self.is_bounded[chosen_variable]:
                    if pertubated_variable < self.initial_ranges[chosen_variable][0]:
                        pertubated_variable = self.initial_ranges[chosen_variable][0]
                    elif pertubated_variable > self.initial_ranges[chosen_variable][1]:
                        pertubated_variable = self.initial_ranges[chosen_variable][1]
                
                candidate_solution = np.array(self.current_solution)
                candidate_solution[chosen_variable] = pertubated_variable
                candidate_solution[-1] = self.cost_function(candidate_solution, chosen_variable)
                
                # Decide if solution will replace the current one based on the Metropolis sampling algorithm
                delta_J = candidate_solution[-1] - self.current_solution[-1] 
                if delta_J < 0:
                    acceptance_probability = 1.0
                    if candidate_solution[-1] < self.best_solution[-1]:
                        self.best_solution = np.array(candidate_solution)
                else:
                    acceptance_probability = math.exp(-delta_J/self.temperature)
                
                if np.random.rand() <= acceptance_probability:
                    if candidate_solution[chosen_variable] != pertubated_variable:
                        print("BIG FUCKING ERROR")
                        exit(-2)
                    self.current_solution[chosen_variable] = candidate_solution[chosen_variable]
                    self.current_solution[-1] = candidate_solution[-1]
        
        return self.best_solution
