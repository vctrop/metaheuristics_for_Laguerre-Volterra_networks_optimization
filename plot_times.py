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
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# x = np.arange(3)
# x_ticks = ("SA", "PSO", "ACO" + r"$_\mathbb{R}$")
# y_ticks = [3600 * i for i in range(7)]
# finite_times = [16682.398, 17564.746, 17203.593]
# infinite_times = [24165.497, 25198.063, 24881.647]

# def hours(x, pos):
    # 'The two args are the value and tick position'
    # hours = int(x / 3600)
    # minutes = int((x % 3600) / 60)
    # #return '%d:%d h' % (hours, minutes)
    # return '0%d h' % hours
    
# formatter = FuncFormatter(hours)

# fig, ax = plt.subplots()
# ax.yaxis.set_major_formatter(formatter)
# plt.hlines(finite_times, xmin=-0.175, xmax=1.975, color='#003366', linestyle='dashed')
# plt.hlines(infinite_times, xmin=0.025, xmax=2.175, color='#339933', linestyle='dashed')
# plt.bar(x-0.1, finite_times  , width=0.15, color='#003366', align='center', label = "Finite order")
# plt.bar(x+0.1, infinite_times, width=0.15, color='#339933', align='center', label = "Infinite order")
# plt.legend(loc="upper left", bbox_to_anchor=(0.65,0.88), fontsize = 16)
# plt.ylabel('Average optimization times', fontsize=18)
# plt.xticks(x, x_ticks)
# plt.yticks(y_ticks)
# ax.tick_params(axis='both', which='major', labelsize=20)
# ax.tick_params(axis='both', which='minor', labelsize=20)


# plt.show()

# THIS IS IMPLEMENTATION-DEPENDENT, DO NOT DISPLAY IN THE PAPER!

def hours(x, pos):
    'The two args are the value and tick position'
    hours = int(x / 3600)
    minutes = int((x % 3600) / 60)
    return '%d:%d h' % (hours, minutes)
    #return '0%d h' % hours
formatter = FuncFormatter(hours)

x = np.arange(3) + 1
x_ticks = ("SA", "PSO", "ACO" + r"$_\mathbb{R}$")


for system_order in ["finite", "infinite"]:
    fig, ax = plt.subplots()
      
    ax.yaxis.set_major_formatter(formatter)
    box_data = []
    for algorithm in ["sa", "pso", "acor"]:
        # Load data
        filename = "./results/" + algorithm + "_" + system_order + "_times.npy"
        train_times     = np.load(filename)
        box_data.append(train_times)
    plt.boxplot(box_data, showfliers=False)
    plt.xticks(x, x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    
plt.show()




