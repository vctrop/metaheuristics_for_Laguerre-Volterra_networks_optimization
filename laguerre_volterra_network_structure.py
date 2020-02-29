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

# Class defining structure of the Laguerre-Volterra network (LVN) for a generic set of parameters
class LVN:
    def __init__(self):
        # Structural parameters
        self.L = None           # laguerre_order
        self.H = None           # num_hidden_units
        self.Q = None           # polynomial_order
        self.T = None           # sampling_interval

        
    # Define order of laguerre filter-bank, number of hidden layer units and polynomial activation order
    def define_structure(self, laguerre_order, num_hidden_units, polynomial_order, sampling_interval):
        self.L = laguerre_order
        self.H = num_hidden_units
        self.Q = polynomial_order
        self.T = sampling_interval
        
        
    # Normalize hidden unit input weights to unit norm and scale polynomial coefficients according to the hidden unit it belongs and the polynomial order
    def normalize_scale_parameters(self, hidden_units_weights, polynomial_coefficients):
        # Shape of the dependent parameters are defined by structural parameters 
        if np.shape(hidden_units_weights) != (self.H, self.L):
            print("Error, wrong shape of hidden unit weights")
            exit(-1)  
        if np.shape(polynomial_coefficients) != (self.H, self.Q):
            print("Error, wrong shape of polynomial coefficients")
            exit(-1)
        
        # Update weights of each hidden unit
        #normalized_weights = hidden_units_weights
        #units_absolute_values = [0.0] * self.H
        normalized_weights = []
        units_absolute_values = []
        for unit_weights in hidden_units_weights:
            #unit_weights = np.array(hidden_units_weights[unit_index * self.L : (unit_index + 1) * self.L - 1])
            #units_absolute_values[unit_index] = math.sqrt( np.sum(unit_weights ** 2) )
            unit_weights = np.array(unit_weights)
            units_absolute_values.append( math.sqrt(np.sum(unit_weights ** 2)) )
            normalized_weights.append( list(unit_weights / units_absolute_values[-1]) )
        
        print("Absolute values")
        print(units_absolute_values)
        
        # Update coefficients of each order
        units_absolute_values = np.array(units_absolute_values)
        scaled_coefficients = np.array(polynomial_coefficients)
        for poly_order in range(1, self.Q + 1):
            scaled_coefficients[:, poly_order - 1] *= (units_absolute_values ** poly_order)
            print(scaled_coefficients[:, poly_order - 1])
            # order_indices = [i for i in range(poly_order, self.Q * self.H, self.Q)]
            # order_coefficients = np.array(polynomial_coefficients)[order_indices]
            # scaled_coefficients[order_indices] = order_coefficients * (units_absolute_values ** poly_order)
        
        return list(normalized_weights), list(scaled_coefficients)
    
        
    # Compute output from imput time-series for a given set of dependent continuous parameters (smoothing constant, filterbank-nonlinearities weights, polynomial coefficients and output offset)
    def compute_output(self, x, laguerre_alpha, hidden_units_weights, polynomial_coefficients, output_offset, weights_modified):
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
        
        # Pre-compute alpha square root to avoid its repeated computation
        alpha_sqrt = np.sqrt(laguerre_alpha)
        filter_bank_outputs = [0] * self.L
        delayed_filter_bank_outputs = list(filter_bank_outputs)                              # look for a way to eliminate the delayed filter bank outputs (it is only really needed in the last term of filter bank outputs computation)
        
        # Input x into the system and get its output y
        y = [output_offset] * len(x)
        for i, sample in enumerate(x):    
            ## update all L laguerre filters outputs
            # update v_0
            delayed_filter_bank_outputs[0] = filter_bank_outputs[0]
            filter_bank_outputs[0] = alpha_sqrt * delayed_filter_bank_outputs[0] +  self.T * np.sqrt(1 - laguerre_alpha) * sample
            # update v_1 .. v_{L-1
            for j in range(1, self.L):
                delayed_filter_bank_outputs[j] = filter_bank_outputs[j]
                filter_bank_outputs[j] = alpha_sqrt * delayed_filter_bank_outputs[j] + alpha_sqrt * filter_bank_outputs[j - 1] - delayed_filter_bank_outputs[j - 1]
            #print(np.array(filter_bank_outputs))
            
            # Compute and accumulate hidden units outputs
            for h in range(self.H):
                # compute hidden unit input via inner product between filter-bank outputs and weights[:, h]
                weighted_inputs = sum([w * v for w, v in zip(list( np.array(hidden_units_weights)[h, :] ), filter_bank_outputs)])
                
                # compute hidden unit output from polynomial coefficients
                # future: Horner's algorithm without constant constant term
                unit_output = 0.0
                for q in range(1, self.Q):
                    unit_output += np.array(polynomial_coefficients)[h, q - 1] * (weighted_inputs ** q)
                
                y[i] += unit_output
        
        return y
        
        
#
def laguerre_filter_memory(alpha):
    M = (-30 - math.log(1 - alpha)) / math.log(alpha)
    M = math.ceil(M)
    
    return M
    

    
    
    
    