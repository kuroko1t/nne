#MIT License
#
#Copyright (c) 2020 kurosawa
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import time
import matplotlib.pyplot as plt

class Benchmark:
    """
    This class is for measuring inference time
    """
    def __init__(self, counter=10, name="sample"):
        self.ave = []
        self.counter = counter
        self.saves = []
        self.name = name

    def measure(self, func, name):
        def inner(*args, **kwargs):
            durations = []
            for i in range(self.counter):
                start = time.time()
                func(*args, **kwargs)
                end = time.time()
                durations.append((end - start) * 1000)
            ave = sum(durations) / self.counter
            self.ave.append(ave)
            min_value = min(durations)
            max_value = max(durations)
            print(f"{name},average[ms],{round(ave, 4)},min[ms],{round(min_value, 4)},max[ms],{round(max_value, 4)}")
            return func(*args, **kwargs)
        return inner


class Plot:
    """
    Take the Benchmark class as an argument and plot the inference time.
    The x-axis assumes batch size.
    """
    def __init__(self, benchmarks:list):
        self.benchmarks = benchmarks

    def plot(self, x, xlabel, title, savefile):
        for bench in self.benchmarks:
            plt.plot(x, bench.ave, "-o", label=bench.name)
        plt.title(title)
        plt.legend()
        plt.xlabel("batch size")
        plt.ylabel("inference time[ms]")
        plt.savefig(savefile)
