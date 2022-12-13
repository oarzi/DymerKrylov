import dimers_util 
import numpy as np
import matplotlib.pyplot as plt
import struct
import sys
import os
import scipy.sparse as sparse
import time
import pickle
from importlib import reload
reload(dimers_util)
from dimers_util import *
from multiprocessing import Pool

def main():
    L = 100
    times = 250
    nums = 150
    d = [45, 55, 65 ,75, 85, 95]
    parallel_analysis(L, times, d, nums)

if __name__ == '__main__':
	main()

