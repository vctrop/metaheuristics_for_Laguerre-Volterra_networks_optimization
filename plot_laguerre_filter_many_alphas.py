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
# from scipy.special import binom
# Own
import laguerre_volterra_network_structure

if len(sys.argv) != 2:
    print("Error, specify the filter order j\n");
    exit(-1)
    
j = int(sys.argv[1])

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

alphas = [0.2, 0.4, 0.6, 0.8]
styles = ['-','--','-.',':']
labels = ['alpha = ' + str(i) for i in alphas]

plt.plot(np.zeros(100), color = 'k')
for alpha, style, label in zip(alphas, styles, labels):
    plt.plot(laguerre_filter_response(j, alpha, 100), label = label, linestyle = style, linewidth=2, color='k')    

plt.ylabel(r'$b_%d[m]$' % j)
plt.xlabel('m')
plt.xlim((0, 100))
plt.legend()
plt.show()