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
import time
# Own
import laguerre_volterra_network_structure
import optimization_utilities
import data_handling
# Third party
import numpy as np

signals_filename = './signals_and_systems/finite_order_train.csv'
lvn_filename = './signals_and_systems/finite_order_train_system.LVN'

train_input, train_output = data_handling.read_io(signals_filename)
# print(np.array(train_input))
# print(np.array(train_output))

Fs = 25
L = 5; H = 3; Q = 4
alpha, W, C, offset = data_handling.read_LVN_file(lvn_filename)

# print(alpha)
# print(W)
# print(C)
# print(offset)

solution_system = laguerre_volterra_network_structure.LVN()
solution_system.define_structure(L, H, Q, 1/Fs)
# solution_output = solution_system.compute_output(train_input, alpha, W, C, offset, False)
# solution_output_mod = solution_system.compute_output_mod(train_input, alpha, W, C, offset, False)

# print('CSV output')
# print(np.array(train_output))
# print('Propagated output')
# print(np.array(solution_output))
# print('Propagated output MOD')
# print(np.array(solution_output))

times_std = []
times_mod = []

for _ in range(10):
    time_start = time.process_time()
    solution_output = solution_system.compute_output(train_input, alpha, W, C, offset, False)
    time_end = time.process_time()
    times_std.append(time_end-time_start)
    
print(f'Times standard = {np.mean(times_std)} ({np.std(times_std)})')