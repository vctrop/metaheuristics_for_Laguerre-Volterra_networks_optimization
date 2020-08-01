#!python3

# Copyright (C) 2020 Victor O. Costa

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

# 3rd party
import numpy as np
import csv
# Own
import simulated_systems
import optimization_utilities

# Write a given LVN structure and system into a file 
def write_LVN_file(file_name, system_parameters):
    print(system_parameters)
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
        
        
# Generate IO data using a Gaussian White Noise (GWN) signal as input to enable the system to capture dynamics of frequency cross-terms, adding GWN to output to reach a certain SNR
def generate_io(system_type, num_samples, file_name, deterministic_parameters):

    if system_type.lower() != "lvn" and system_type.lower() != "cascade":
        print("The system type must be \"lvn\" or \"cascade\"")
        exit(-1)
    
    # Unit is a zero mean and unit variance Gaussian white noise (GWN) signal
    input = np.random.normal(0.0, 1.0, num_samples)
    # Finite order
    if system_type == "lvn":
        L = 5; H = 3; Q = 4
        # Train
        if deterministic_parameters == None:
            noiseless_output, deterministic_parameters = simulated_systems.simulate_LVN_random(input, L, H, Q)
        # Test
        else:
            noiseless_output = simulated_systems.simulate_LVN_deterministic(input, L, H, Q, deterministic_parameters)
            
        write_LVN_file(file_name, deterministic_parameters)
    # Infinite order
    else:
        # Train
        if deterministic_parameters == None:
            noiseless_output, alphas = simulated_systems.simulate_cascaded_random(input, 3)
        # Test
        else:
            noiseless_output = simulated_systems.simulate_cascaded_deterministic(input, alphas)
        
    # Output additive Gaussian White Noise 
    SNR_db = 5                                  # Signal-to-Noise ratio in decibels
    out_avg_pwr = np.mean(np.array(noiseless_output) ** 2)          # Average power of output signal
    out_avg_pwr_db = 10 * np.log10(out_avg_pwr)                           
    # As SNR_db = sig_power_db - noise_power_db, noise_power_db = sig_power_db - SNR_db
    noise_avg_pwr_db = out_avg_pwr_db - SNR_db
    noise_avg_pwr = 10 ** (noise_avg_pwr_db / 10)
    # For a GWN signal X, the average power is equal to the second moment E[X^2] = mean^2 + std^2. With zero mean, the average power is equal to std^2, the variance
    GWN_std = np.sqrt(noise_avg_pwr)
    noise = np.random.normal(0.0, GWN_std, num_samples)
    
    # Generate noisy output
    output = noiseless_output + noise
    
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