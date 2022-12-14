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
from itertools import repeat

@dataclass
class Simulator:
    L : int
    times: int
    d : list
    nums : int
    prob : int = 1
    local: bool = True
    d_procs_num : int = 1
    batch_subprocs_num : int = 1
    save : bool = True
    analysis_rhos : list = None
    

    def progress_bar(self, iterable):
        if self.local:
            tqdm_text = "#" + ("{}->".format(os.getppid()) + "{}".format(os.getpid())).ljust(12) + " "
            return tqdm(iterable, miniters= self.times//25, desc=tqdm_text) 
        else:
            return iterable
    
    def parallel_analysis(self):
        self.analysis_rhos = []
        H = {'H_ring' : get_h_ring(self.L), 'H_hopp' : get_h_hop(self.L)}
        print("Starting {}, {} | {}".format(len(H['H_ring']), len(H['H_hopp']), time.strftime("%Y_%m_%d__%H_%M")))
        manager = Manager()
        queue = manager.Queue()
        sema = Semaphore(self.d_procs_num)
        ds = self.d.copy()
        procs = []
        while ds:
            sema.acquire()
            curr_d = ds.pop(0)
            p = Process(target=self.classical_evolution, args=(H, curr_d, queue, sema),daemon=False)
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
        
        fname = 'analysis_L{}_t{}_d{}'.format(self.L, self.times, time.strftime("%Y_%m_%d__%H_%M"))
        self.analysis_rhos = [fname]
        while not queue.empty():
            self.analysis_rhos.append(queue.get())
        
        if self.save:
            with open("analyses/" + fname + ".pickle", 'wb') as handle:
                pickle.dump(self.analysis_rhos, handle)
        print("Finished parallel_analysis for  L =  {}, times = {}, d = {}, nums = {} | {}".format(self.L, self.times, 
                                                                                        self.d, self.nums, 
                                                                                        time.strftime("%Y_%m_%d__%H_%M")))
        return self.analysis_rhos
    
    def classical_evolution(self, H, _d, q, sema):
        print("id: {}, L =  {}, times = {}, d = {}, nums = {}".format(os.getpid(), self.L, self.times, _d, self.nums))
        with Pool(self.batch_subprocs_num) as p:
            c_rhos =  p.starmap(self.classical_evolutions_batch, repeat((H, _d), self.batch_subprocs_num), chunksize=1)
            p.close()
            p.join()
        
        rho = np.array(c_rhos)
        print("before batch sum", rho.shape)
        rho = np.sum(rho, axis=0)/self.nums
        print("after batch sum", rho.shape)
        
        analyzed = self.analyze(rho)
        analyzed['d'] = _d
        q.put(analyzed, False)     
        print("{} finished.".format(os.getpid())) 
        sema.release()
        return 0

    
    def classical_evolutions_batch(self, H, _d):
        H_ring, H_hopp, = H['H_ring'], H['H_hopp']

        p_array = np.concatenate((np.ones(len(H_ring)),self.prob*np.ones(len(H_hopp))))/(len(H_ring)+self.prob*len(H_hopp))
        allgates = H_ring + H_hopp

        initial_psi = [get_initial_config(self.L, _d)]*(self.nums//self.batch_subprocs_num)
        psi = np.array(initial_psi, dtype=np.int32)
        # print("psi0.shape=", psi.shape)
        
        rho = np.apply_along_axis(defect_density, 1 , psi)
        rho = np.sum(rho, axis=0).reshape(1, self.L)
        
        # print("rho0.shape=", rho.shape)

        def apply(f):
            return f[0](f[1])

        for i in self.progress_bar(range(self.times)):
            if not self.local and (i % (self.times//25) == 0):
                print("{}->{} is  {}% completed".format(os.getppid(), os.getpid(), 100*i/times), flush=True)
            gates_i = np.random.choice(allgates, size=self.nums, p=p_array)
            psi = np.array(list(map(apply, zip(gates_i, psi))))
            # print("psi.shape=", psi.shape)
            charge = np.apply_along_axis(defect_density, 1 , psi)
            # print("charge.shape=", charge.shape)
            rho = np.vstack((rho, np.sum(np.apply_along_axis(defect_density, 1 , psi), axis=0).reshape(1, self.L)))

            # psi = np.vstack((psi, [psi_next]))
            # print("rho.shape=", rho.shape)

        
        # print("{}->{} before= {}".format(os.getppid(), os.getpid(), psi.shape))
        # print("{} batch rho={}".format(os.getppid(), rho.shape))

        return rho

    def analyze(self, rho):
        print("Analysis start")
        analysis = {}
        analysis['rho'] = rho
        analysis['Median'] = 1 + np.sum((np.cumsum(rho[:,1:],axis=1)<0.5).astype(int),axis=1).reshape(rho.shape[0])
        sites = [np.arange(1, rho.shape[1])]

        analysis['Mean'] = np.average(np.repeat(sites,rho.shape[0],axis=0), axis=1, weights=rho[:, 1:]).reshape(analysis['Median'].shape)
        analysis['std'] = np.sqrt(np.average((np.repeat(sites,rho.shape[0],axis=0) -                        analysis['Mean'].reshape(rho.shape[0],1))**2 , axis=1, weights=rho[:, 1:])).reshape(analysis['Median'].shape)
        analysis['speed'] = analysis['Mean'][1:] - analysis['Mean'][:-1]
        analysis['acc'] = analysis['speed'][1:] - analysis['speed'][:-1]
        print("Analysis end")
        return analysis
    
def plot_analysis(analysis, L, times, nums, save=False):

    analysis_rep = analysis[1:]
    fig, ax = plt.subplots(3, gridspec_kw={'height_ratios':[3, 1, 1]}, figsize=(13, 10))
    # fig.suptitle('L={}, times={}, nums={}'.format(L, times, nums))
    
    for a in analysis_rep:
        ax[0].plot(a['Mean'], label=a['d'])
    ax[0].legend()
    ax[0].set_title("Mean position")

    for a in analysis_rep:
        ax[1].plot(a['speed'], label=a['d'])
    ax[1].set_title("Speed")

    for a in analysis_rep:
        ax[2].plot(a['acc'], label=a['d'])
    ax[2].set_title("acceleration")
    
    fig.tight_layout()
    if save:
        plt.savefig("figs/" + analysis[0], format='png')
    plt.show()

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
