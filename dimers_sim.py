import pickle
from dimers_util import *
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

@dataclass
class Experiment: 
    file_name : str
    dir_name : str
    results : list
    description : str = ''
    
    def save(self):
        with open(self.dir_name + self.file_name + ".pickle", 'wb') as f:
            pickle.dump(self, f)

@dataclass
class Analysis:   
    L : int
    times: int
    d : int
    nums : int
    rho : np.ndarray
    label : str
    file_name : str
    dir_name : str
    analysis: dict = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.analyze()
        
    def save(self):
        with open(self.dir_name + self.file_name + ".pickle", 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(self.dir_name + self.file_name + ".pickle", 'rb') as f:
            return pickle.load(f)

    def analyze(self):
        print("Analysis start")
        self.analysis['d'] = self.d
        self.analysis['rho'] = self.rho
        self.analysis['nums'] = self.nums
        self.analysis['times'] = self.times
        self.analysis['L'] = self.L
        self.analysis['label'] = self.label
        
        self.analysis['Median'] = 1 + np.sum((np.cumsum(self.rho[:,1:],axis=1)<0.5).astype(int),axis=1).reshape(self.rho.shape[0])
        sites = [np.arange(1, self.rho.shape[1])]

        self.analysis['Mean'] = np.average(np.repeat(sites,self.rho.shape[0],axis=0), axis=1, weights=self.rho[:, 1:]).reshape(self.analysis['Median'].shape)
        self.analysis['std'] = np.sqrt(np.average((np.repeat(sites, self.rho.shape[0], axis=0) -                        self.analysis['Mean'].reshape(self.rho.shape[0], 1))**2 , axis=1, weights=self.rho[:, 1:])).reshape(self.analysis['Median'].shape)
        self.analysis['speed'] = self.analysis['Mean'][1:] - self.analysis['Mean'][:-1]
        self.analysis['acc'] = self.analysis['speed'][1:] - self.analysis['speed'][:-1]
        print("Analysis end")
        return self.analysis
    
@dataclass
class Simulator:
    L : int
    times: int
    d : int
    nums : int
    
    prob : int = 1
    file_name : str = ""
    dir_name : str = "analyses/"
    batch_subprocs_num : int = 1
    save : bool = False
    local: bool = True
    analysis : Analysis = None

    def __post_init__(self):
        self.file_name = self.file_name if self.file_name else 'analysis_L{}_t{}_n{}_d{}___'.format(self.L, self.times, 
                                                                          self.nums, self.d,   
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
        while simulators:
            sema.acquire()
            sim = simulators.pop(0)
            p = Process(target=Simulator.simulate_parallel_task, args=(sim, queue, sema),daemon=False)
            procs.append(p)
            p.start()
        
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
        q.put(sim.simulate())
        sema.release()
        return 0
    
    
    def simulate(self):
        print("Starting id: {}, L =  {}, times = {}, d = {}, nums = {} |".format(os.getpid(), self.L, self.times, self.d, self.nums, time.strftime("%Y_%m_%d__%H_%M")))
        
        H = {'H_ring' : get_h_ring(self.L), 'H_hopp' : get_h_hop(self.L)}
        with Pool(self.batch_subprocs_num) as p:
            c_rhos =  p.map(self.classical_evolutions_batch, (H for i in range(self.batch_subprocs_num)), chunksize=1)
            p.close()
            p.join()
        
        rho = np.array(c_rhos)
        print("before batch sum", rho.shape)
        rho = np.sum(rho, axis=0)/self.nums
        print("after batch sum", rho.shape)

        analysis = Analysis(L=self.L, times=self.times, d=self.d, nums=self.nums, rho=rho, label='nums', file_name = self.file_name, dir_name=self.dir_name)
        
        if self.save:
            analysis.save()
            
        print("Finished id {}: L =  {}, times = {}, d = {}, nums = {} | {}".format(os.getpid(), self.L, self.times, 
                                                                                        self.d, self.nums, 
                                                                                        time.strftime("%d_%m_%Y__%H_%M")))
        return analysis

    def classical_evolutions_batch(self, H):
        H_ring, H_hopp, = H['H_ring'], H['H_hopp']

        p_array = np.concatenate((np.ones(len(H_ring)),self.prob*np.ones(len(H_hopp))))/(len(H_ring)+self.prob*len(H_hopp))
        allgates = H_ring + H_hopp

        initial_psi = [get_initial_config(self.L, self.d)]*(self.nums//self.batch_subprocs_num)
        psi = np.array(initial_psi, dtype=np.int32)
        # print("psi0.shape=", psi.shape)
        
        rho = np.apply_along_axis(defect_density, 1 , psi)
        rho = np.sum(rho, axis=0).reshape(1, self.L)
        
        def apply(f):
            return f[0](f[1])

        for i in self.progress_bar(range(self.times)):
            if not self.local and (i % (self.times//25) == 0):
                print("{}->{} is  {}% completed".format(os.getppid(), os.getpid(), 100*i/self.times), flush=True)
            gates_i = np.random.choice(allgates, size=self.nums, p=p_array)
            psi = np.array(list(map(apply, zip(gates_i, psi))))
            # print("psi.shape=", psi.shape)
            charge = np.apply_along_axis(defect_density, 1 , psi)
            # print("charge.shape=", charge.shape)
            rho = np.vstack((rho, np.sum(np.apply_along_axis(defect_density, 1 , psi), axis=0).reshape(1, self.L)))

            # psi = np.vstack((psi, [psi_next]))
        print("rho.shape=", rho.shape)

        if not self.local:
            print("{}->{} finished".format(os.getppid(), os.getpid(), flush=True))

        return rho
    

    
def plot_analyses(analyses,save=False, title='', name=''):
    lwdt = 1

    fig, ax = plt.subplots(3, gridspec_kw={'height_ratios':[1, 1, 1]}, figsize=(13, 10))
    if title:
        fig.suptitle(title)
    
    for a in analyses:
        ax[0].plot(a.analysis['Mean'], label=a.analysis[a.analysis['label']], linewidth=lwdt)
    ax[0].legend()
    ax[0].set_title("Mean position")

    for a in analyses:
        ax[1].plot(a.analysis['speed'], label=a.analysis[a.analysis['label']], linewidth=lwdt)
    ax[1].set_title("Speed")

    for a in analyses:
        ax[2].plot(a.analysis['acc'], label=a.analysis[a.analysis['label']], linewidth=lwdt)
    ax[2].set_title("acceleration")
    
    fig.tight_layout()
    if save and name:
        plt.savefig("figs/" + path + '.png', format='png')
    plt.show()
    
def plot_dist(rh):
    return

def plot_rho(analysis,c=False):
    plt.figure(figsize=[16,12])
    plt.pcolor(analysis['rho'], cmap='binary')
    
    y = range(analysis['rho'].shape[0])
    plt.plot(analysis['Median'], y, 'b-', linewidth=2, label="Median")
    plt.plot(analysis['Mean'], y, color='lime', linestyle='-', linewidth=1, label="Mean")
    
    plt.fill_betweenx(y, analysis['Mean'] - analysis['std'], analysis['Mean'] + analysis['std'],
                 color='darkgreen', alpha=0.2, label="std")


    plt.xlabel('$x$')
    plt.ylabel('$T$', rotation=0)
    plt.colorbar()
    plt.legend()
    plt.show()
