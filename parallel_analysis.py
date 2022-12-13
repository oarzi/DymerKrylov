import dimers_util 
import dimers_sim
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
    d_sim = [55, 65 ,75, 85, 95]
    L_sim = 100
    times_sim = 5000
    nums_sim = 10000
    d_procs_sim = 5
    nums_subprocs_sim = 10
    simulator = dimers_sim.Simulator(L=L_sim, times=times_sim, d=d_sim, nums=nums_sim, d_procs_num=d_procs_sim, nums_subprocs_num = nums_subprocs_sim)
    simulator.parallel_analysis()
    dimers_sim.plot_analysis(simulator.analysis_rhos, d_sim, L_sim, times_sim, nums_sim)
if __name__ == '__main__':
	main()

