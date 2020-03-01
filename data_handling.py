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
import csv
import simulated_systems
import optimization_utilities

# Write a given LVN structure and system into a file 
def write_LVN_file(file_name, system_parameters):
    L = len(system_parameters[1][0])
    H = len(system_parameters[1])
    Q = len(system_parameters[2][0])
    system_file_name = file_name + "_system.LVN"
    with open(system_file_name, mode = 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([L, H, Q])
        for parameters in system_parameters:
            writer.writerow((np.array(parameters)).flatten())
    
    
# Reads LVN file and returns the system's parameters
def read_LVN_file(file_name):
    with open(file_name, mode = 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        csv_strings = []
        for row in csv_reader:
            csv_strings.append(row)
        
        L, H, Q   = list( np.array(csv_strings[0]).astype(np.int) )
        
        alpha       = float(csv_strings[1][0])
        flat_W      = list( np.array(csv_strings[2]).astype(np.float) )
        flat_C      = list( np.array(csv_strings[3]).astype(np.float) )
        offset      = float(csv_strings[4][0])
        
        concatenated_parameters = [alpha] + flat_W + flat_C + [offset]
        alpha, W, C, offset = optimization_utilities.decode_solution(concatenated_parameters, L, H, Q)
        
        return alpha, W, C, offset
        
        
# Generate IO data using a Gaussian White Noise (GWN) signal as input to enable the system to capture dynamics of frequency cross-terms
def generate_io(system_type, num_samples, file_name, deterministic_parameters):
    input = np.random.standard_normal(size = num_samples)
    
    if system_type == "lvn":
        L = 5; H = 3; Q = 4
        if deterministic_parameters == None:
            output, deterministic_parameters = simulated_systems.simulate_LVN_random(input, L, H, Q)
        else:
            output = simulated_systems.simulate_LVN_deterministic(input, L, H, Q, deterministic_parameters)
            
        write_LVN_file(file_name, deterministic_parameters)
    else:
        #output = simulated_systems.simulate_trig_exp(input)
        exit(-1)
        
    csv_name = file_name + ".csv"
    with open(csv_name, mode = 'w', newline='') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(input)
        csv_writer.writerow(output)
    
    return deterministic_parameters

# Read IO data from CSVs
def read_io(file_name):
    input = []
    output = []
    
    with open(file_name, mode = 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        csv_strings = []
        for row in csv_reader:
            csv_strings.append(row)
        
        input_string = csv_strings[0]
        output_string = csv_strings[1]
        
        for index in range(len(input_string)):
            input.append(float(input_string[index]))
            output.append(float(output_string[index]))
        
    return input, output