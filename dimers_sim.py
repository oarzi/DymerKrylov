import dimers_util
import dimers_analysis
from importlib import reload
reload(dimers_analysis)
reload(dimers_util)
import pickle
from multiprocessing import Pool, Queue, Process, Semaphore, Manager
from scipy.sparse.linalg import expm_multiply
import time
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    pass
import os
from dataclasses import dataclass, field
import argparse

import sys

import numpy as np
import lzma
    
@dataclass
class Simulator:
    L : int
    times: int
    d : int
    
    batch : int = 1
    gate : object = dimers_util.Gate2
    prob : int = 0.5
    check_interval: int = 100
    from_file : bool = False
    
    
    file_name : str = field(default="", init=True)
    dir_name : str = "analyses/"
    batch_procs_num : int = 1
    local: bool = True
    
    def __post_init__(self):
        if not self.file_name:
            self.file_name = 'analysis_L{}_d{}_t{}___'.format(self.L, self.d, self.check_interval*self.times,  
                                                              time.strftime("%Y_%m_%d__%H_%M"))
    
    def progress_bar(self, iterable):
        if self.local:
            tqdm_text = "#" + ("{}->".format(os.getppid()) + "{}".format(os.getpid())).ljust(12) + " "
            return tqdm(iterable, miniters= self.times//25, desc=tqdm_text, position=0, leave=False) 
        else:
            return iterable
    
    @classmethod
    def simulate_parallel(cls, simulators, procs_num):
        manager = Manager()
        queue = manager.Queue()
        sema = Semaphore(procs_num)
        procs = []
        jobs = len(simulators)
        print("simulate_parallel # jobs = {}".format(jobs))
        count = 0
        while simulators:
            sema.acquire()
            sim = simulators.pop(0)
            p = Process(target=Simulator.simulate_parallel_task, args=(sim, queue, sema),daemon=False)
            procs.append(p)
            p.start()
            count += 1
            print("count = {}".format(count))
            
        if count is not jobs:
            raise ValueError
        
        print("Waiting for all processes to close")
        print("{} items waiting".format(queue.qsize()))
        while procs:
            curr_p = procs.pop(0)
            if curr_p.is_alive():
                curr_p.join()
                curr_p.close()

        print("All processes closed")
        print("{} items waiting".format(queue.qsize()))
        
        results = []
        while not queue.empty():
            results.append(queue.get())


        print("Finished simulate_parallel for {} simulations | {}".format(jobs, 
                                                                                        time.strftime("%d_%m_%Y__%H_%M")))
        return results
    
    @classmethod
    def simulate_parallel_task(cls, sim, q, sema):
        result = sim.simulate()
        q.put(result)
        sema.release()
        return 0
    
    def initialize(self):
        if self.from_file:
            with lzma.open(self.dir_name[:-1] + "psis/" + self.file_name + "_psi.pickle", 'rb') as f:
                psis = pickle.load(f)               
            rho = dimers_analysis.Analysis.load(self.dir_name + self.file_name + ".pickle").rho
        else:
            psis = [dimers_util.get_initial_config_point(self.L, self.d, self.batch)]*self.batch_procs_num
            rho = np.mean(dimers_util.defect_density_point(psis[0]), axis=0).reshape((1, self.L))
            
        print(rho.shape)
        return rho, psis
    
    
    def get_H(self):
        H_ring = np.array([self.gate(i, False) for i in range(0, self.L - 1)], dtype=object)
        H_hop = np.array([self.gate(i, True, False if i < (self.L - 2) else True)  for i in range(0, self.L - 1)],
                         dtype=object)
        
        return (H_ring, H_hop)
    
    def simulate(self):
        print("Starting id: {}, L =  {}, # times = {}, d = {}, #batch = {} , # of batches = {} | {}".format(
            os.getpid(), self.L, self.check_interval*self.times, self.d, self.batch, self.batch_procs_num, 
            time.strftime("%Y_%m_%d__%H_%M")))
        
        H = self.get_H()
        rho, psi = self.initialize()
        
        analysis = dimers_analysis.Analysis(L=self.L, times=self.check_interval*self.times, d=self.d, batch=self.batch,
                                            p=self.prob, rho=rho, psis=[], file_name = self.file_name, 
                                            dir_name=self.dir_name)

        for i in self.progress_bar(range(self.check_interval)):
            if not self.local and (i % (self.check_interval//25) == 0):
                print("======================================================================================")
                print("{} is {}% completed".format(os.getpid(), 100*i/self.check_interval), flush=True)
            rho, psi = self.simulation_iteration(rho, psi, H)

            analysis.rho = rho
            analysis.save()  
            with lzma.open(self.dir_name[:-1] + "psis/" + self.file_name + "_psi.pickle", "wb", preset=9) as f:
                pickle.dump(psi, f)
            
        print("Finished id {}: L =  {}, # times = {}, d = {}, # batch = {} | {}".format(os.getpid(), self.L,
                                                                                        self.check_interval*self.times, 
                                                                                        self.d, self.batch, 
                                                                                        time.strftime("%d_%m_%Y__%H_%M")))
        return analysis
    
    def simulation_iteration(self, rho, psis, H):
        H_ring, H_hop = H
        with Pool(self.batch_procs_num) as p:
            c_rhos = p.starmap(self.classical_evolutions_batch_points, 
                               [(psi, rho[-1], H_ring, H_hop) for psi in psis])
            rhos, psis = [res[0] for res in c_rhos], [res[1] for res in c_rhos]

            print("before batch sum ({},{})".format(len(c_rhos), rhos[0].shape))
            print("psi shape {}".format(psis[0].shape))

            rho = np.vstack((rho, np.mean(rhos, axis=0)))

            print("after batch sum", rho.shape)
            p.close()   
        return rho, psis
    
    def classical_evolutions_batch_points(self, psi, rho, H_ring, H_hop):
        
        for i in range(self.times):
            psi = dimers_util.promote_psi_classical(psi, H_ring, H_hop, self.prob)
            charge = dimers_util.defect_density_point(psi)
            rho = np.vstack((rho, np.mean(charge, axis=0)))

        return rho[1:], psi
    
@dataclass
class QuantumSimulator(Simulator):
    def initialize(self):
        if self.from_file:
            with lzma.open(self.dir_name[:-1] + "psis/" + self.file_name + "_psi.pickle", 'rb') as f:
                psis = pickle.load(f)               
            rho = dimers_analysis.Analysis.load(self.dir_name + self.file_name + ".pickle").rho
        else:
            configs = dimers_util.load_configs(self.L)
            psi = dimers_util.get_initial_config_point_quantum(self.L, self.d, configs)
            rho = np.array([dimers_util.defect_density_points_quantum(configs,psi)])
            
        print(rho.shape)
        return rho, psi
    
    def get_H(self):
        configs = dimers_util.load_configs(self.L)
        H = dimers_util.load_data(self.L)
        H_ring, H_hop = H["H_ring"], H["H_hopp"]
        
        return (H_ring, H_hop, configs)
    
    def simulation_iteration(self, rho, psi, H):

        H_ring, H_hop, configs = H

        for i in range(self.times):
            psi = expm_multiply(-1j*(1 - self.prob)*H_ring, psi)
            psi = expm_multiply(-1j*self.prob*H_hop, psi)
            rho = np.vstack((rho, dimers_util.defect_density_points_quantum(configs,psi)))
            
        return rho, psi
    
def test_rand(n):
    with Pool(5) as p:
        c_rhos =  p.map(print_rand, (n for _ in range(n)), chunksize=1)
        p.close()
        p.join()
        
def print_rand(i):
    for _ in range(3):
        print(np.random.randint(0, 100, i))

def get_experiment_args():
    parser = argparse.ArgumentParser(prog='Parallel execution of experiments and their analysis.', allow_abbrev=False)
    subparsers = parser.add_subparsers(help='Choose experiment', required=True, dest='experiment')

    parser_quantum = subparsers.add_parser('q', help='Quantum simulation', allow_abbrev=False)
    
    parser_quantum.add_argument("--L", help="System size.", type=int, nargs=1,  required=True)
    parser_quantum.add_argument("--times", help="Number of time steps.", type=int, nargs=1, required=True)
    parser_quantum.add_argument("--check", help="Number interval checkpoints", type=int, nargs=1,
                                                   required=True)
    parser_quantum.add_argument("--batch", help="Number of trajectories over which path is averaged.", type=int,
                                           nargs='+', required=False, default=[1])
    parser_quantum.add_argument("--p", help="Probability for hoping gate", type=float, nargs='+', default=[0.5])
    parser_quantum.add_argument("--d", help="Defect's inital location.", type=int, nargs=1, required=True)

    parser_quantum.add_argument("--name", help="File prefix",
                                           type=str, nargs='+', default='q_')
    parser_quantum.add_argument("--procs_sim", help="Number of simultaneously running experiments", type=int, required=False,
                                           nargs=1, default=[1])
    parser_quantum.add_argument("--batch_procs", help="Number of processes per single running experiment",
                                           type=int, nargs='+', default=[1])
    
    
    parser_varying_batch_size = subparsers.add_parser('bs', help='Varying batch size experiment', allow_abbrev=False)
    
    parser_varying_batch_size.add_argument("--L", help="System size.", type=int, nargs=1,  required=True)
    parser_varying_batch_size.add_argument("--times", help="Number of time steps.", type=int, nargs=1, required=True)
    parser_varying_batch_size.add_argument("--d", help="Defect's inital location.", type=int, nargs=1, required=True)
    
    parser_varying_batch_size.add_argument("--batch", help="Number of trajectories over which path is averaged.", type=int,
                                           nargs='+', required=True)
    parser_varying_batch_size.add_argument("--procs_sim", help="Number of simultaneously running experiments", type=int,
                                           nargs=1, default=1)
    parser_varying_batch_size.add_argument("--batch_procs", help="Number of processes per single running experiment",
                                           type=int, nargs='+', default=[1])
    
    parser_varying_batch_size.add_argument("--name", help="File prefix",
                                           type=str, nargs='+', default='bs')
    
    parser_varying_initial_conditions = subparsers.add_parser('ic', help='Varying varying initial conditions experiment',
                                                              allow_abbrev=False)
    
    parser_varying_initial_conditions.add_argument("--L", help="System size.", type=int, nargs=1,  required=True)
    parser_varying_initial_conditions.add_argument("--times", help="Number of time steps.", type=int, nargs=1,
                                                   required=True)
    parser_varying_initial_conditions.add_argument("--d", help="Defect's inital location.", type=int, nargs='+',
                                                   required=True)
    parser_varying_initial_conditions.add_argument("--p", help="Probability for hoping gate", type=float, nargs='+',
                                                   required=True, default=[0.5])
    parser_varying_initial_conditions.add_argument("--batch", help="Number of trajectories over which path is averaged.",
                                                   type=int, nargs=1, required=True)
    parser_varying_initial_conditions.add_argument("--procs_sim", help="Number of simultaneously running experiments",
                                                   type=int, nargs=1, default=1)
    parser_varying_initial_conditions.add_argument("--batch_procs", help="Number of processes per single running experiment", type=int, nargs='+', default=1)
    
    parser_varying_initial_conditions.add_argument("--name", help="File prefix",
                                           type=str, nargs='+', default='ic')
    
    parser_varying_prob = subparsers.add_parser('pgate', help='Varying gate probabilities',
                                                              allow_abbrev=False)
    
    parser_varying_prob.add_argument("--L", help="System size.", type=int, nargs=1,  required=True)

    parser_varying_prob.add_argument("--times", help="Number of time steps per interval.", type=int, nargs=1,
                                                   required=True)
                                                   
    parser_varying_prob.add_argument("--check", help="Number interval checkpoints", type=int, nargs=1,
                                                   required=True)
                                                   
    parser_varying_prob.add_argument("--d", help="Defect's inital location.", type=int, nargs=1,
                                                   required=True)
   
    parser_varying_prob.add_argument("--p", help="Probability for hoping gate", type=float, nargs='+',
                                                   required=True)
                                                   
    parser_varying_prob.add_argument("--batch", help="Number of trajectories over which path is averaged.",
                                                   type=int, nargs=1, required=True)
    parser_varying_prob.add_argument("--procs_sim", help="Number of simultaneously running experiments",
                                                   type=int, nargs=1, default=1)
    parser_varying_prob.add_argument("--batch_procs", help="Number of processes per single running experiment", type=int, nargs='+', default=1)
    
    parser_varying_prob.add_argument("--name", help="File prefix",
                                           type=str, nargs='+', default='pgate_')

    return parser
