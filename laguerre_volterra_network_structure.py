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

# Class defining structure of the Laguerre-Volterra network (LVN) for a generic set of parameters
class LVN:
    def __init__(self):
        # Structural parameters
        self.laguerre_order = None          # L
        self.num_hidden_units = None        # H
        self.polynomial_order = None        # Q
        self.sampling_interval = None       # Fs

        
    # Define order of laguerre filter-bank, number of hidden layer units and polynomial activation order
    def define_structure(self, laguerre_order, num_hidden_units, polynomial_order, sampling_interval):
        self.laguerre_order = laguerre_order
        self.num_hidden_units = num_hidden_units
        self.polynomial_order = polynomial_order
        self.sampling_interval = sampling_interval

        
    # Compute output from imput time-series for a given set of dependent continuous parameters (smoothing constant, filterbank-nonlinearities weights, polynomial coefficients and output offset)
    def compute_predictions(self, x, laguerre_alpha, hidden_unit_weights, polynomial_coefficients, output_offset):
        ## Error checking
        # Network structure must be specified before dependent parameters
        if self.laguerre_order == None or self.num_hidden_units == None or self.polynomial_order == None:
            print("Error, first define the LVN structure")
            exit(-1)
        # Laguerre filterbank smoothing constant must be between 0 and 1
        if laguerre_alpha < 0 or laguerre_alpha > 1:
            print("Error, invalid laguerre alpha")
            exit(-1)
        # Shape of the dependent parameters are defined by structural parameters 
        if np.shape(hidden_unit_weights) != (self.laguerre_order, self.num_hidden_units):
            print("Error, wrong shape of hidden unit weights")
            exit(-1)  
        if np.shape(polynomial_coefficients) != (self.polynomial_order, self.num_hidden_units):
            print("Error, wrong shape of polynomial coefficients")
            exit(-1)
        
        # Pre-compute alpha square root to avoid its repeated computation
        alpha_sqrt = np.sqrt(laguerre_alpha)
        
        # 
        filter_bank_outputs = [0] * self.laguerre_order
        delayed_filter_bank_outputs = list(filter_bank_outputs)                              # look for a way to eliminate the delayed filter bank outputs (it is only really needed in the last term of filter bank outputs computation)
        
        # Input x into the system and get its output y
        y = [output_offset] * len(x)
        for i, sample in enumerate(x):    
            ## update all L laguerre filters outputs
            # update v_0
            delayed_filter_bank_outputs[0] = filter_bank_outputs[0]
            filter_bank_outputs[0] = alpha_sqrt * delayed_filter_bank_outputs[0] + self.sampling_interval * np.sqrt(1 - laguerre_alpha) * sample
            # update v_1 .. v_{L-1}
            for j in range(1, self.laguerre_order):
                delayed_filter_bank_outputs[j] = filter_bank_outputs[j]
                filter_bank_outputs[j] = alpha_sqrt * delayed_filter_bank_outputs[j] + alpha_sqrt * filter_bank_outputs[j - 1] - delayed_filter_bank_outputs[j - 1]
            
            # Compute and accumulate hidden units outputs
            for h in range(self.num_hidden_units):
                # compute hidden unit input via inner product between filter-bank outputs and weights[:, h]
                weighted_inputs = sum([w * v for w, v in zip(list( np.array(hidden_unit_weights)[:, h] ), filter_bank_outputs)])
                
                # compute hidden unit output from polynomial coefficients
                # future: Horner's algorithm without constant constant term
                unit_output = 0.0
                for q in range(1, self.polynomial_order):
                    unit_output += np.array(polynomial_coefficients)[q - 1, h] * (weighted_inputs ** q)
                
                y[i] += unit_output
        
        return y