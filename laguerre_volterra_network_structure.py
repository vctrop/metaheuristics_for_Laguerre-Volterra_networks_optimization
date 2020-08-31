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

# Python std lib
import math
from collections.abc import Iterable
# Third party
import numpy as np



class LVN:
    ''' Class defining structure of the Laguerre-Volterra network (LVN) for a generic set of parameters. '''
    def __init__(self):
        ''' Constructor. '''
        # Structural parameters
        self.L = None           # laguerre_order
        self.H = None           # num_hidden_units
        self.Q = None           # polynomial_order
        self.T = None           # sampling_interval

        
    def define_structure(self, laguerre_order, num_hidden_units, polynomial_order, sampling_interval):
        ''' Define order of laguerre filter-bank, number of hidden layer units and polynomial activation order. '''
        self.L = laguerre_order
        self.H = num_hidden_units
        self.Q = polynomial_order
        self.T = sampling_interval
        
      
    def normalize_scale_parameters(self, hidden_units_weights, polynomial_coefficients):
        ''' Normalize hidden unit input weights to unit norm and scale polynomial coefficients according to the hidden unit it belongs and the polynomial order. '''
        # Shape of the dependent parameters are defined by structural parameters 
        if np.shape(hidden_units_weights) != (self.H, self.L):
            print("Error, wrong shape of hidden unit weights")
            exit(-1)  
        if np.shape(polynomial_coefficients) != (self.H, self.Q):
            print("Error, wrong shape of polynomial coefficients")
            exit(-1)
        
        # Update weights of each hidden unit
        normalized_weights = []
        units_absolute_values = []
        for unit_weights in hidden_units_weights:
            unit_weights = np.array(unit_weights)
            units_absolute_values.append( math.sqrt(np.sum(unit_weights ** 2)) )
            normalized_weights.append( list(unit_weights / units_absolute_values[-1]) )

        # Update coefficients of each order
        units_absolute_values = np.array(units_absolute_values)
        scaled_coefficients = np.array(polynomial_coefficients)
        for poly_order in range(1, self.Q + 1):
            scaled_coefficients[:, poly_order - 1] *= (units_absolute_values ** poly_order)
            
        return list(normalized_weights), list(scaled_coefficients)
        
        
    def propagate_laguerre_filterbank(self, signal, alpha):
        ''' Propagate input signal through the Laguerre filter bank.
            The output is an (L,N) matrix. '''
        
        # Sanity check
        if not isinstance(signal, Iterable):
            print('Error, input signal must be an iterable object')
            exit(-1)
        if alpha <= 0:
            print('Error, alpha must be positive')
            exit(-1)
        
        alpha_sqrt = math.sqrt(alpha)
        bank_outputs = np.zeros((self.L, 1 + len(signal)))      # The bank_outputs matrix initially has one extra column to represent zero values at n = -1
        
        # Propagate V_{j} with j = 0
        for n, sample in enumerate(signal):
            bank_outputs[0, n + 1] = alpha_sqrt * bank_outputs[0, n - 1 + 1] +  self.T * np.sqrt(1 - alpha) * sample
        
        # Propagate V_{j} with j = 1, .., L-1
        for j in range(1, self.L):
            for n in range(len(signal)):
                bank_outputs[j, n + 1] = alpha_sqrt * (bank_outputs[j, n - 1 + 1] + bank_outputs[j - 1, n + 1]) - bank_outputs[j - 1, n - 1  + 1]
        
        bank_outputs = bank_outputs[:,1:]
        
        return bank_outputs
        
        
    def compute_output(self, x, laguerre_alpha, hidden_units_weights, polynomial_coefficients, output_offset, weights_modified):
        ''' Compute output from input time-series for a given set of dependent continuous parameters (smoothing constant, filterbank-nonlinearities weights, polynomial coefficients and output offset). '''
        ## Error checking
        # Network structure must be specified before dependent parameters
        if self.L == None or self.H == None or self.Q == None:
            print("Error, first define the LVN structure")
            exit(-1)
        # Laguerre filterbank smoothing constant must be between 0 and 1
        if laguerre_alpha < 0 or laguerre_alpha > 1:
            print("Error, invalid laguerre alpha")
            exit(-1)
        # Shape of the dependent parameters are defined by structural parameters 
        if np.shape(hidden_units_weights) != (self.H, self.L):
            print("Error, wrong shape of hidden unit weights")
            exit(-1)  
        if np.shape(polynomial_coefficients) != (self.H, self.Q):
            print("Error, wrong shape of polynomial coefficients")
            exit(-1)
        
        if weights_modified:
            hidden_units_weights, polynomial_coefficients = self.normalize_scale_parameters(hidden_units_weights, polynomial_coefficients)
        
        # Precompute alpha square root to avoid repeated computation
        alpha_sqrt = np.sqrt(laguerre_alpha)
        hidden_units_weights = np.array(hidden_units_weights)
        polynomial_coefficients = np.array(polynomial_coefficients)
        
        # Propagate the input signal through the filter bank
        # Filter bank outputs mat is (L, N)
        N = len(x)
        laguerre_outputs = self.propagate_laguerre_filterbank(x, laguerre_alpha)
        
        # Define the input of each hidden node as the dot product between the Laguerre filterbank outputs and the weight vectors
        # Hidden nodes inputs mat is (N,H)
        hidden_nodes_inputs = np.matmul(laguerre_outputs.T, hidden_units_weights.T)
        
        # The outputs of hidden layer mat is (N, HQ+1).
        # Each node has one projection as input and Q values as outputs (Q-1 of them are nonlinear)
        # All positions of the first column are ones to account for the output offset
        hidden_layer_out = np.ones((N, self.H * self.Q + 1))
        for q in range(1, self.Q + 1):
            hidden_layer_out[:, 1  + (q - 1) * self.H : 1 + q * self.H] = np.power(hidden_nodes_inputs, q)
        
        # Flatten polynomial coefficients to compute the final output from hidden layer outputs using matrix-vector multiplication
        flattened_coefficients = (polynomial_coefficients.T).flatten()
        # print(flattened_coefficients)
        # print(output_offset)
        # The output offset in the first position is always multiplied by 1
        linear_params = np.concatenate(([output_offset], flattened_coefficients))
        
        
        y = hidden_layer_out @ linear_params
        
        return y
        
        
def laguerre_filter_memory(alpha):
    ''' Rough estimate of the extent of significative values in the Laguerre bank's impulse responses. '''
    M = (-30 - math.log(1 - alpha)) / math.log(alpha)
    M = math.ceil(M)
    
    return M
    

    
    
    
    