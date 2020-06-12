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
import math

class LVN:
    ''' Class defining structure of the Laguerre-Volterra network (LVN) for a generic set of parameters '''
    def __init__(self):
        ''' Constructor '''
        # Structural parameters
        self.L = None           # laguerre_order
        self.H = None           # num_hidden_units
        self.Q = None           # polynomial_order
        self.T = None           # sampling_interval


    def define_structure(self, laguerre_order, num_hidden_units, polynomial_order, sampling_interval):
        ''' Define order of Laguerre filter-bank, number of hidden layer units and polynomial activation order '''
        self.L = laguerre_order
        self.H = num_hidden_units
        self.Q = polynomial_order
        self.T = sampling_interval
        
        
    def normalize_scale_parameters(self, hidden_units_weights, polynomial_coefficients):
        ''' Normalize hidden unit input weights to unit norm and scale polynomial coefficients according to the hidden unit it belongs and the polynomial order '''
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
    
    
    def propagate_laguerre_filterbank(self, x, laguerre_alpha):
        '''  '''
        # Error checking
        if self.L == None or self.H == None or self.Q == None:
            print("Error, first define the LVN structure")
            exit(-1)
        if laguerre_alpha < 0 or laguerre_alpha > 1:
            print("Error, invalid Laguerre alpha")
            exit(-1)
        
        # Pre-compute alpha to avoid repeated computation
        alpha_sqrt = math.sqrt(laguerre_alpha)
        # Define arrays for the filterbank outputs of a sample from x
        sample_outputs = [0] * self.L
        delayed_sample_outputs = list(sample_outputs)             # look for a way to eliminate the delayed filter bank outputs (it is only really needed in the last term of filter bank outputs computation)
        # Filterbank outputs for all x
        filterbank_outputs = []
        # 
        for sample in x:
            ## Update all L Laguerre filters outputs
            # update v_0
            delayed_sample_outputs[0] = sample_outputs[0]
            sample_outputs[0] = alpha_sqrt * delayed_sample_outputs[0] +  self.T * np.sqrt(1 - laguerre_alpha) * sample
            # update v_1 .. v_{L-1
            for j in range(1, self.L):
                delayed_sample_outputs[j] = sample_outputs[j]
                sample_outputs[j] = alpha_sqrt * delayed_sample_outputs[j] + alpha_sqrt * sample_outputs[j - 1] - delayed_sample_outputs[j - 1]
            
            filterbank_outputs.append(sample_outputs[:])
            
        return filterbank_outputs
        
    
    def polynomial_activation(self, x, filterbank_outputs, hidden_units_weights, polynomial_coefficients, output_offset, weights_modified):
        '''  '''
        ## Error checking
        # Network structure must be specified before dependent parameters
        if self.L == None or self.H == None or self.Q == None:
            print("Error, first define the LVN structure")
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

        # Compute system outputs y based on the outputs of the Laguerre filterbank
        y = [0] * len(x)
        for i in range(len(y)):    

            # Compute and accumulate hidden units outputs
            for h in range(self.H):

                # compute hidden unit input via inner product between filter-bank outputs and weights[:, h]
                weighted_inputs = sum([w * v for w, v in zip(list( np.array(hidden_units_weights)[h, :] ), filterbank_outputs[i])])
                
                # compute hidden unit output from polynomial coefficients
                # future: Horner's algorithm without constant constant term
                unit_output = 0.0
                for q in range(1, self.Q):
                    unit_output += np.array(polynomial_coefficients)[h, q - 1] * (weighted_inputs ** q)
                
                y[i] += unit_output
            # Account for constant offset
            y[i] += output_offset
        return y
        

    def propagate_LVN(self, x, laguerre_alpha, hidden_units_weights, polynomial_coefficients, output_offset, weights_modified):
        ''' Compute output from input time-series for a given set of dependent continuous parameters (smoothing constant, filterbank outputs weights, polynomial coefficients and output offset) '''
        
        filterbank_outputs = self.propagate_laguerre_filterbank(x, laguerre_alpha)
        # print(np.array(filterbank_outputs))
        y = self.polynomial_activation(x, filterbank_outputs, hidden_units_weights, polynomial_coefficients, output_offset, weights_modified)
        
        return y

def laguerre_filter_memory(alpha):
    M = (-30 - math.log(1 - alpha)) / math.log(alpha)
    M = math.ceil(M)
    
    return M
    

    
    
    
    