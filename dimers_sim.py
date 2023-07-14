from dimers_util import *
from dimers_analysis import *
import pickle
from multiprocessing import Pool, Queue, Process, Semaphore, Manager
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt
import time
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    pass
import os
from dataclasses import dataclass, field
import argparse
import sys
    
@dataclass
class Simulator:
    L : int
    times: int
    d : int
    
    batch : int = 1
    gate : object = Gate2
    prob : int = 0.5
    check_interval: int = 100
    from_file : bool = False
    
    
    file_name : str = ""
    dir_name : str = "analyses/"
    batch_procs_num : int = 1
    local: bool = True
    analysis : Analysis = None

    def __post_init__(self):
        self.file_name = self.file_name if self.file_name else 'analysis_L{}_t{}_b{}_d{}___'.format(self.L, self.times, 
                                                                          self.batch, self.d,   
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
            search = "L{}_d{}_p{}".format(self.L, self.d, self.prob)
            path = next(x for x in os.listdir(self.dir_name) if search in x)
            with open(self.dir_name  + path, 'rb') as f:
                ana = pickle.load(f)
                rho, psis = ana.rho, ana.psis
        else:
            psis = [get_initial_config_point(self.L, self.d, self.batch)]*self.batch_procs_num
            rho = np.mean(defect_density_point(psis[0]), axis=0).reshape((1, self.L))
            
        print(rho.shape)
        return rho, psis
    
    
    
    def simulate(self):
        print("Starting id: {}, L =  {}, # times = {}, d = {}, #batch = {} , # of batches = {} | {}".format(os.getpid(), self.L, self.times, self.d, self.batch, self.batch_procs_num, time.strftime("%Y_%m_%d__%H_%M")))
        
        H_ring = np.array([self.gate(i, False) for i in range(0, self.L - 1)], dtype=object)
        H_hop = np.array([self.gate(i, True, False if i < self.L -2 else True)  for i in range(0, self.L - 1)], dtype=object)
        
        rho, psis = self.initialize()

        for i in range(self.check_interval):
            with Pool(self.batch_procs_num) as p:
                c_rhos = p.starmap(self.classical_evolutions_batch_points, 
                               [(psis[i], rho[-1], H_ring, H_hop) for i in range(self.batch_procs_num)])
                rhos, psis = [res[0] for res in c_rhos], [res[1] for res in c_rhos]

                print("before batch sum ({},{})".format(len(c_rhos), rhos[0].shape))
                print("psi shape {}".format(psis[0].shape))
                
                rho = np.vstack((rho, np.mean(rhos, axis=0)))

                print("after batch sum", rho.shape)

                analysis = Analysis(L=self.L, times=self.times, d=self.d, batch=self.batch, p=self.prob, rho=rho,
                                    psis=psis, file_name = self.file_name, dir_name=self.dir_name)
                analysis.save()
            
        print("Finished id {}: L =  {}, # times = {}, d = {}, # batch = {} | {}".format(os.getpid(), self.L, self.times, 
                                                                                        self.d, self.batch, 
                                                                                        time.strftime("%d_%m_%Y__%H_%M")))
        return analysis
    
    def classical_evolutions_batch_points(self, psi, rho, H_ring, H_hop):
        
        for i in self.progress_bar(range(self.times)):
            if not self.local and (i % (self.times//25) == 0):
                print("{}->{} is  {}% completed".format(os.getppid(), os.getpid(), 100*i/self.times), flush=True)
            
            psi = promote_psi_classical(psi, H_ring, H_hop, self.prob)
            charge = defect_density_point(psi)
            rho = np.vstack((rho, np.mean(charge, axis=0)))

        print("{}->{} finished".format(os.getppid(), os.getpid(), flush=True))

        return rho[1:], psi
    

    def quantum_evolutions_batch_points(self, dt=0.5):
        H = load_data(self.L)
        H_ring, H_hop, configs = H["H_ring"], H["H_hopp"], H["configs"]

        psi = get_initial_config_point_quantum(self.L, self.d, configs)

        rho = np.array([defect_density_points_quantum(configs,psi)])

        for i in  self.progress_bar(range(self.times)):
            if not self.local and (i % (self.times//25) == 0):
                print("{}->{} is  {}% completed".format(os.getppid(), os.getpid(), 100*i/self.times), flush=True)
            psi = expm_multiply(-1j*(self.prob*H_ring + (1 - self.prob)*H_hop)*dt,psi)
            rho = np.vstack((rho, defect_density_points_quantum(configs,psi)))

        analysis = Analysis(L=self.L, times=self.times, d=self.d, batch=self.batch, p=self.prob, rho=rho, file_name = self.file_name, dir_name=self.dir_name)
        
        analysis.save()
            
        print("Finished id {}: L =  {}, # times = {}, d = {}, # batch = {} | {}".format(os.getpid(), self.L, self.times, 
                                                                                        self.d, self.batch, 
                                                                                        time.strftime("%d_%m_%Y__%H_%M")))
        return analysis
    

    def classical_evolutions_batch(self, H):
        H_ring, H_hopp, = H['H_ring'], H['H_hopp']
        p_array = np.concatenate((np.ones(len(H_ring)),self.prob*np.ones(len(H_hopp))))/(len(H_ring)+self.prob*len(H_hopp))

        allgates = H_ring + H_hopp

        initial_psi = [get_initial_config(self.L, self.d)]*(self.batch//self.batch_procs_num)
        psi = np.array(initial_psi, dtype=np.int32)
        # print("psi0.shape=", psi.shape)
        
        charge = np.apply_along_axis(defect_density, 1 , psi)
        rho = np.sum(charge, axis=0)
        
        def apply(f):
            return f[0](f[1])

        for i in self.progress_bar(range(self.times)):
            if not self.local and (i % (self.times//25) == 0):
                print("{}->{} is  {}% completed".format(os.getppid(), os.getpid(), 100*i/self.times), flush=True)
            rng = np.random.default_rng()
            gates = rng.choice(allgates, size=self.batch//self.batch_procs_num, p=p_array)
            psi = np.array(list(map(apply, zip(gates_i, psi))))
            # print("psi.shape=", psi.shape)
            charge = np.apply_along_axis(defect_density, 1 , psi)
            charge0 = charge[:,0]
            if np.sum(charge0) !=  psi.shape[0]:
                with open("bad_matrix{}.txt".format(os.getpid()), "w") as f:
                    np.set_printoptions(threshold=sys.maxsize)
                    f.write(str(np.sum(charge0)) + "\n")
                    f.write(str(charge0.shape) + "\n")
                    f.write(str(psi.shape) + "\n")
                    f.write(str(np.argwhere(charge0 != 1)) + "\n")
                    f.write("========================================\n")
                    f.write("Charge0: \n" + str(charge0) + "\n")                                
                    f.write("========================================\n")
                    f.write("Bad Charge:\n" + str(charge[np.argwhere(charge0 != 1)]) + "\n")
                    f.write("========================================\n")
                    f.write("Gates:\n" + str(gates_i[np.argwhere(charge0 != 1)]))
                raise ValueError()
            # print("charge.shape=", charge.shape)
            rho = np.vstack((rho, np.sum(charge, axis=0)))

            # psi = np.vstack((psi, [psi_next]))
        print("rho.shape=", rho.shape)

        if not self.local:
            print("{}->{} finished".format(os.getppid(), os.getpid(), flush=True))

        return rho
    
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
    parser_quantum.add_argument("--batch", help="Number of trajectories over which path is averaged.", type=int,
                                           nargs='+', required=False, default=[1])
    parser_quantum.add_argument("--p", help="Probability for hoping gate", type=float, nargs='+', default=[0.5])
    parser_quantum.add_argument("--d", help="Defect's inital location.", type=int, nargs=1, required=True)

    parser_quantum.add_argument("--name", help="File prefix",
                                           type=str, nargs='+', default='1')
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

    parser_varying_prob.add_argument("--times", help="Number of time steps.", type=int, nargs=1,
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
                                           type=str, nargs='+', default='pgate')

    return parser
