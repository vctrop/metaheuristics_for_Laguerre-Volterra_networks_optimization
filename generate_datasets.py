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

import data_handling

## Generate train and test data
# Finite-order system
# Train
train_system_parameters = data_handling.generate_io("lvn", 1024, "finite_order_train", None)
# Test
data_handling.generate_io("lvn", 2048, "finite_order_test", train_system_parameters)

# Infinite-order system
# Train
train_alphas = data_handling.generate_io("cascade", 1024, "infinite_order_train", None)
# Test
data_handling.generate_io("cascade", 1024, "infinite_order_test", train_alphas)