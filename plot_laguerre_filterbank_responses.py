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
import sys
import math
# 3rd party
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
# Own
import laguerre_volterra_network_structure

if len(sys.argv) != 3:
    print("Error, specify the filterbank order {L} and the {alpha} parameters\n");
    exit(-1)
    
L = int(sys.argv[1])
alpha = float(sys.argv[2])

def binom(n, k):
    if n < k:
        return 0
    
    return math.factorial(n) / ( math.factorial(k) * math.factorial(n - k))

def laguerre_filter_response(order, alpha, memory):
    filter_impulse_response = []
    for m in range(memory):
        sum = 0.0
        for k in range(order+1):                
            sum += ((-1) ** k) * binom(m, k) * binom(order, k) * (alpha ** (order - k)) * ((1 - alpha) ** k)

        output = (alpha ** ((m - order)/2)) * np.sqrt(1 - alpha) * sum
        filter_impulse_response.append(output)
    
        
    return np.array(filter_impulse_response)
    
def laguerre_filterbank_response(max_order, alpha, memory):
    filterbank_impulse_responses = []
    for j in range(max_order):
        order_response = laguerre_filter_response(j, alpha, memory)
        filterbank_impulse_responses.append(order_response)
        
    return np.array(filterbank_impulse_responses)
    
    
# #memory = laguerre_volterra_network_structure.laguerre_filter_memory(alpha)
laguerre_bank = laguerre_filterbank_response(L, alpha, 250)

plt.figure(figsize=(10,10))
# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


labels = ['j = %d' % j for j in range(L)]
   
plt.plot(np.zeros(len(laguerre_bank[0,:])), color = 'k')
for j in range(L):
    plt.plot(laguerre_bank[j, :], label = labels[j], linewidth=2)

plt.ylabel(r'$b_j[m]$')
plt.xlabel('m')
plt.xlim((0, 250))
plt.legend()
plt.show()