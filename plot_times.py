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
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

x = np.arange(3)
x_ticks = ("SA", "PSO", "ACO" + r"$_\mathbb{R}$")
finite_times = [16682.398, 17564.746, 17203.593]
infinite_times = [24165.497, 25198.063, 24881.647]

def hours(x, pos):
    'The two args are the value and tick position'
    hours = int(x / 3600)
    minutes = int((x % 3600) / 60)
    return '%d:%d h' % (hours, minutes)

formatter = FuncFormatter(hours)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.bar(x-0.1, finite_times  , width=0.15, color='#003366', align='center', label = "Finite order")
plt.bar(x+0.1, infinite_times, width=0.15, color='#339933', align='center', label = "Infinite order")
plt.legend(loc="upper left", bbox_to_anchor=(0.7,1))
plt.xticks(x, x_ticks)

plt.show()