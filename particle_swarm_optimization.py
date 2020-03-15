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
from base_metaheuristic import Base
    
class PSO(Base):
    """ Class for the Adaptative Inertia Weight Particle Swarm Optimization (AIWPSO), following (Nickabadi et al., 2011) """

    def __init__(self):
        """ Constructor """
        # I nitial algorithm parameters
        self.num_iter = 0
        self.population_size = 0
        self.personal_weight = 0.5
        self.global_weight = 0.5
        #self.inertia_max = 1.0
        #self.inertia_min = 0.0

        # Optimization results
        self.swarm_positions = None
        self.swarm_velocities = None
        self.personal_bests = None
        self.global_best = None
        
        
    def set_parameters(self, num_iter, population_size, personal_weight, global_weight):
        """ Define values for the parameters used by the algorithm """
        # Input error checking
        if num_iter <= 0:
            print("Number of iterations must be greater than zero")
            exit(-1)
        if population_size <= 0:
            print("Population size must be greater than zero")
            exit(-1)
        if personal_weight < 0 or global_weight < 0:
            print("Personal and global weights must be equal or greater than zero")
            exit(-1)
            
        self.num_iter = num_iter
        self.population_size = population_size
        self.personal_weight = personal_weight
        self.global_weight = global_weight
        
    
    def define_variables(self, initial_ranges, is_bounded):
        """ Defines the number of variables, their initial values ranges and wether or not these ranges constrain the variable during the search """
        # Input error checking
        if self.num_iter == 0:
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
        
        self.swarm_positions = np.zeros((self.population_size, self.num_variables + 1))
        self.swarm_velocities = np.zeros((self.population_size, self.num_variables))
        
        self.personal_bests = np.zeros((self.population_size, self.num_variables + 1))
        self.global_best = np.zeros(self.num_variables + 1)
        self.global_best[-1] = float('inf')
            

    def optimize(self):
        """ Initializes the archive and enter the main loop, until it reaches maximum number of iterations """
        # Variables and cost function must be defined prior to optimization
        if self.num_variables == None:
            print("Error, number of variables and their boundaries must be defined prior to optimization")
            exit(-1)
        if self.cost_function == None:
            print("Error, cost function must be defined prior to optimization")
            exit(-1)
        
        # Initialize swarm positions and velocities randomly (population_size cost function evaluations)
        for i in range(self.population_size):
            for j in range(self.num_variables):
                self.swarm_positions[i, j] = np.random.uniform(self.initial_ranges[j][0], self.initial_ranges[j][1])
                self.swarm_velocities[i, j] = np.random.uniform(self.initial_ranges[j][0], self.initial_ranges[j][1])
            self.swarm_positions[i, -1] = self.cost_function(self.swarm_positions[i, :-1], -1)
            # Update global best
            if self.swarm_positions[i, -1] < self.global_best[-1]:
                self.global_best = self.swarm_positions[i, :] 
        
        self.personal_bests = np.array(self.swarm_positions)
        
        # Main optimization loop (population_size * num_iter cost function evaluations)
        for iteration in range(self.num_iter):
            for particle in range(self.population_size):
                # Update velocity vector
                self.swarm_velocities[particle, :] =    (self.swarm_velocities[particle, :]
                                                        + self.personal_weight  * np.random.rand() * (self.personal_bests[particle, :-1]    - self.swarm_positions[particle, :-1])
                                                        + self.global_weight    * np.random.rand() * (self.global_best[:-1]                 - self.swarm_positions[particle, :-1]))
                # Update position vector
                self.swarm_positions[particle, :-1] = self.swarm_positions[particle, :-1] + self.swarm_velocities[particle, :]
                # Restrict search for bounded variables
                for var in range(self.num_variables):
                    if self.is_bounded[var]:
                        # Use the hard border strategy
                        if self.swarm_positions[particle, var] < self.initial_ranges[var][0]:
                            self.swarm_positions[particle, var] = self.initial_ranges[var][0]
                        elif self.swarm_positions[particle, var] > self.initial_ranges[var][1]:
                            self.swarm_positions[particle, var] = self.initial_ranges[var][1]        
                
                # Compute cost of new position
                self.swarm_positions[particle, -1] = self.cost_function(self.swarm_positions[particle, :-1], -1)
                # Update personal best solution
                if self.swarm_positions[particle, -1] < self.personal_bests[particle, -1]:
                    self.personal_bests[particle, :] = self.swarm_positions[particle, :]
                    # Update global best solution
                    if self.personal_bests[particle, -1] < self.global_best[-1]:
                        self.global_best = self.personal_bests[particle, :]
                
        
        return self.global_best