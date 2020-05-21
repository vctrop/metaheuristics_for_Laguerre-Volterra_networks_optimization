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
    
# Python standar lib
import math
# 3rth party
import numpy as np
from scipy.stats import norm
# Own
from base_metaheuristic import Base

class ACOr(Base):
    """ Class for the Ant Colony Optimization for Continuous Domains, following (Socha and Dorigo, 2006) """

    def __init__(self):
        """ Constructor """
        # Define verbosity and NULL problem definition
        super().__init__()
        
        # Initial algorithm parameters
        self.relative_iterations = None                 # Array containing the iterations at which best solutions are reported
        self.num_iter = 0                               # Number of iterations
        self.pop_size = 5                               # Population size
        self.k = 50                                     # Archive size
        self.q = 0.01                                   # Locality of search (selection of pivot ants)
        self.xi = 0.85                                  # Speed of convergence (spreadness of ant generation)
        
        # Optimization results
        self.SA = None                                  # Solution Archive
        self.best_solution = None                       # Best solution of the archive
        

    def set_parameters(self, pop_size, k, q, xi, function_evaluations_array):
        """ Define values for the parameters used by the algorithm """
        # Input error checking
        if len(function_evaluations_array) == 0:
            print("Error, objective function evaluation array must not be empty")
            exit(-1)
        if pop_size <= 0 or k <= 0 or q <= 0 or xi <= 0:
            print("Error, parameters must be non-null positives")
            exit(-1)
            
        
        # Number of function evaluations for ACOr: pop_size * num_iterations
        function_evaluations_array = np.array(function_evaluations_array)
        self.relative_iterations = (function_evaluations_array - k) / pop_size
        all_divisible = (np.array([x.is_integer() for x in self.relative_iterations])).all()
        if not all_divisible:
            print("Error, at least one number of function evaluations subtracted by k is not divisible by population size m")
            exit(-1)
        
        self.num_iter = int(np.max(self.relative_iterations))
        self.pop_size = pop_size
        self.k = k
        self.q = q
        self.xi = xi

    
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
        self.SA = np.zeros((self.k, self.num_variables + 1))

    
    def _biased_selection(self, probabilities):
        """ Returns an index based on a set of probabilities (also known as roulette wheel selection in GA) """
        r = np.random.uniform(0, sum(probabilities))
        for i, f in enumerate(probabilities):
            r -= f
            if r <= 0:
                return i
    

    def update_success_rate(self, success_count):
        """ Success rate is not updated in vanilla ACOr """
        pass
    
    def control_xi(self):
        """ Xi is not updated in vanilla ACOr """
        pass
    
    def control_q(self):
        """ q is not updated in vanilla ACOr """
        pass
    
    def handle_adaptions(self, success_count):
        self.update_success_rate(success_count)
        self.control_q()
        self.control_xi()
    
    def optimize(self):
        """ Initializes the archive and enter the main loop, until it reaches maximum number of iterations """
        # Error checking
        if self.num_variables == None:
            print("Error, number of variables and their boundaries must be defined prior to optimization")
            exit(-1)
        if self.cost_function == None:
            print("Error, cost function must be defined prior to optimization")
            exit(-1)
        
        # Keep solutions defined by function_evaluations_array
        recorded_solutions = []
        
        # Initialize the archive by random sampling, respecting each variable's boundaries   
        if self.verbosity:   print("[INITIALIZING SOLUTION ARCHIVE]")
        pop = np.zeros((self.pop_size, self.num_variables +1))
        w = np.zeros(self.k)
        
        for i in range(self.k):
            for j in range(self.num_variables): 
                self.SA[i, j] = np.random.uniform(self.initial_ranges[j][0], self.initial_ranges[j][1])     # Initialize solution archive randomly
            self.SA[i, -1] = self.cost_function(self.SA[i, 0:self.num_variables], -1)                       # Get initial cost for each solution
        self.SA = self.SA[self.SA[:, -1].argsort()]                                                         # Sort solution archive (best solutions first)
        
        # Array containing indices of solution archive position
        x = np.linspace(1,self.k,self.k) 
        w = norm.pdf(x,1,self.q*self.k)                                         # Weights as a gaussian function of rank with mean 1, std qk
        p = w/sum(w) 
        
        if self.verbosity:   print("ALGORITHM MAIN LOOP")
        # Algorithm runs until it reaches the determined number of iterations
        for iteration in range(self.num_iter):
            if self.verbosity:
                print("[%d]" % iteration)
                print(self.SA[0, :])
            
            success_count = 0                                                   # Count how many ant improve the solution they are sampling from    
            Mi = self.SA[:, 0:self.num_variables]                               # Matrix of means
            for ant in range(self.pop_size):                                    # For each ant in the population
                l = self._biased_selection(p)                                   # Select solution of the SA to sample from based on probabilities p
                
                for var in range(self.num_variables):                           # Calculate the standard deviation of all variables from solution l
                    sigma_sum = 0
                    for i in range(self.k):
                        sigma_sum += abs(self.SA[i, var] - self.SA[l, var])
                    sigma = self.xi * (sigma_sum/(self.k - 1))
                     
                    pop[ant, var] = np.random.normal(Mi[l, var], sigma)         # Sample from normal distribution with mean Mi and st. dev. sigma
                    
                    # Search space boundaries violation is only dealt with when the variable is considered bounded (self.is_bounded)
                    if self.is_bounded[var]:
                        # Use the hard border strategy
                        if pop[ant, var] < self.initial_ranges[var][0]:
                            pop[ant, var] = self.initial_ranges[var][0]
                        elif pop[ant, var] > self.initial_ranges[var][1]:
                            pop[ant, var] = self.initial_ranges[var][1]        
                        
                        # Use the random position strategy
                        # if pop[ant, var] < self.initial_ranges[var][0] or pop[ant, var] > self.initial_ranges[var][1]:                   
                            # pop[ant, var] = np.random.uniform(self.initial_ranges[var][0], self.initial_ranges[var][1])
                    
                # Evaluate cost of new solution
                pop[ant, -1] = self.cost_function(pop[ant, 0:self.num_variables], -1)       
                
                # Check if the new solution is better than the one the ant sampled from
                if pop[ant, -1] < self.SA[l, -1]:
                    success_count += 1
                    
            # Compute success rate, updates xi and q (No effect in vanilla ACOr)
            self.handle_adaptions(success_count)
            
            # Append new solutions to the Archive
            self.SA = np.append(self.SA, pop, axis = 0)                                                         
            # Update PDF from which ants sample their centers, according to updates in q parameter
            w = norm.pdf(x,1,self.q*self.k)                                         # Weights as a gaussian function of rank with mean 1, std qk
            p = w/sum(w)                                                            # Probabilities of selecting solutions as search guides
        
            # Sort solution archive according to the fitness of each solution
            self.SA = self.SA[self.SA[:, -1].argsort()]                                                         
            # Remove worst solutions
            self.SA = self.SA[0:self.k, :]   
            # Extract current best solution
            self.best_solution = self.SA[0, :]
            if (self.relative_iterations - 1 == iteration).any():
                recorded_solutions.append(self.best_solution)
            
        return np.array(recorded_solutions)
        