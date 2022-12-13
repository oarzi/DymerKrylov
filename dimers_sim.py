import pickle
from dimers_util import *
from multiprocessing import Pool, SimpleQueue, Process, Semaphore
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

def quantum_evolution(L, times, H, d=0, J=1):
    H_ring, H_hopp, configs = H['H_ring'], H['H_hopp'], H['configs']
    dim = H_ring.shape[0]
    psi, i0 = get_initial_state(L, configs, dim, d)
    

    dt = 0.5
    rho = [defect_density(configs,psi)]

    for i in  tqdm(range(times)):
        psi = expm_multiply(-1j*dt*H_ring,psi)
        psi = expm_multiply(-1j*dt*J*H_hopp,psi)

        rho.append(defect_density(configs,psi))

    rho = np.array(rho)
    
    return rho

def classical_evolutions_single(L, times, H, d, _p):
    H_ring, H_hopp, configs = H['H_ring'], H['H_hopp'], H['configs']
    
    psi, i = get_initial_state(L, configs, H_ring.shape[0], d)
    rho = np.array([defect_density(configs,psi)])
    
    
    steps = [i]
    for _ in  tqdm(range(times)):
        rings_i = list(H_ring.getrow(i).nonzero()[1])
        hopps_i = list(H_hopp.getrow(i).nonzero()[1])
        p_array = np.concatenate((np.ones(len(rings_i)), _p*np.ones(len(hopps_i))))/(len(rings_i)+_p*len(hopps_i))
        i = np.random.choice(rings_i + hopps_i, p=p_array)
        curr_psi = np.zeros(psi.shape)
        curr_psi[i] = 1.

        rho = np.row_stack((rho ,defect_density(configs,curr_psi)))
        steps.append(i)
    
    return (rho, steps)

def classical_evolutions_single2(L, times, H, d, _p):
    H_ring, H_hopp, = H['H_ring'], H['H_hopp']
    gates = H_ring + H_hopp
    p_array = np.concatenate((np.ones(len(H_ring)), _p*np.ones(len(H_hopp))))/(len(H_ring)+_p*len(H_hopp))
    psi =  get_initial_config(L, d)
    rho = np.array([defect_density(psi)])
    
    gate = np.random.choice(gates, size=times, p=p_array)
    
    for gate in tqdm(gates, miniters=20):
        # gate = np.random.choice(gates, p=p_array)
        psi = gate(psi)
        rho = np.row_stack((rho ,defect_density(psi)))
    return rho

def classical_evolution(L, times, H, d=0, nums=1, steps=False, p=1):
    print("id: {}, L =  {}, times = {}, d = {}, nums = {}".format(os.getpid(), L, times, d, nums))
    start = time.process_time()
    vc = np.vectorize(classical_evolutions_single2, otypes='O', cache=True)
    
    results = vc([L]*nums, [times]*nums, [H]*nums, [d]*nums, [p]*nums)
    
    rhos = np.array(results)
    rhos = np.sum(rhos, axis=0)/nums
    
    end = time.process_time()
    print("Elapsed time during the {} in seconds: {}".format(os.getpid(), end-start)) 
    return (rhos, [res for res in results[1]]) if steps else rhos



def parallel_analysis(L, times, d, nums):
    H_ring, H_hopp = get_h_ring(L), get_h_hop(L)
    print(len(H_ring), len(H_hopp))
    H = {'H_ring' : H_ring, 'H_hopp' : H_hopp}
    with Pool(6) as p:
        c_rhos =  p.starmap(classical_evolutions_single3, ((L, times, H, d, nums) for d in d), chunksize=1)
        p.close()
        p.join()
    analysis_rhos =  [analyze(rho) for rho in c_rhos]
    with open('analysis_L{}_t{}_d{}.pickle'.format(L, times, time.strftime("%Y_%m_%d_%H_%M")), 'wb') as handle:
        pickle.dump(analysis_rhos, handle)
    return analysis_rhos

@dataclass
class Simulator:
    L : int
    times: int
    d : list
    nums : int
    prob : int = 1
    local: bool = True
    d_procs_num : 1
    nums_subprocs_num : 1
    
    def __post_init__(self):
        self.analysis_rhos = []

    def progress_bar(self, iterable):
        return tqdm(iterable, miniters= self.times//100) if self.local else iterable
    
    def parallel_analysis(self):
        H = {'H_ring' : get_h_ring(self.L), 'H_hopp' : get_h_hop(self.L)}
        print(len(H['H_ring']), len(H['H_hopp']))
        queue = SimpleQueue()
        sema = Semaphore(self.d_procs_num)
        ds = self.d.copy()
        procs = []
        while ds:
            sema.acquire()
            curr_d = ds.pop(0)
            p = Process(target=self.classical_evolution, args=(H, curr_d, queue, sema),daemon=False)
            procs.append(p)
            p.start()

        while procs:
            curr_p = procs.pop(0)
            if curr_p.is_alive():
                curr_p.join()
                curr_p.close()
        
        while not queue.empty():
            self.analysis_rhos.append(queue.get())
        queue.close()
        with open('analyses/analysis_L{}_t{}_d{}.pickle'.format(self.L, self.times, time.strftime("%Y_%m_%d_%H_%M")), 'wb') as handle:
            pickle.dump(self.analysis_rhos, handle)
        return self.analysis_rhos
    
    def classical_evolution(self, H, _d, q, sema):
        print("id: {}, L =  {}, times = {}, d = {}, nums = {}".format(os.getpid(), self.L, self.times, self.d, self.nums))
        start = time.process_time()
        with Pool(self.nums_subprocs_num) as p:
            c_rhos =  p.starmap(self.classical_evolutions_nums, ((H, _d) for i in range(2)), chunksize=1)
            p.close()
            p.join()
        
        rho = np.array(c_rhos)
        rho = np.sum(rho, axis=1)/self.nums
        
        analyzed = self.analyze(rho)
        analyzed['d'] = _d
        q.put(analyzed)     
        print("Elapsed time during the {} in seconds: {}".format(os.getpid(), time.process_time() - start)) 
        sema.release()
        return 0

    
    def classical_evolutions_nums(self, H, _d):
        H_ring, H_hopp, = H['H_ring'], H['H_hopp']

        p_array = np.concatenate((np.ones(len(H_ring)),self.prob*np.ones(len(H_hopp))))/(len(H_ring)+self.prob*len(H_hopp))
        allgates = H_ring + H_hopp

        initial_psi = get_initial_config(self.L, _d)
        psi = np.array([[initial_psi]*(self.nums//self.nums_subprocs_num)], dtype=np.int32)

        def apply(f):
            return f[0](f[1])

        for _ in self.progress_bar(range(self.times)):
            gates_i = np.random.choice(allgates, size=self.nums, p=p_array)
            psi_next = np.array(list(map(apply, zip(gates_i, psi[-1]))))
            psi = np.vstack((psi, [psi_next]))

        rho = np.apply_along_axis(defect_density, 2 , psi)
        rho = np.sum(rho, axis=1)
        
        return rho

    def analyze(self, rho):
        analysis = {}
        analysis['rho'] = rho
        analysis['Median'] = 1 + np.sum((np.cumsum(rho[:,1:],axis=1)<0.5).astype(int),axis=1).reshape(rho.shape[0])
        sites = [np.arange(1, rho.shape[1])]

        analysis['Mean'] = np.average(np.repeat(sites,rho.shape[0],axis=0), axis=1, weights=rho[:, 1:]).reshape(analysis['Median'].shape)
        analysis['std'] = np.sqrt(np.average((np.repeat(sites,rho.shape[0],axis=0) -                        analysis['Mean'].reshape(rho.shape[0],1))**2 , axis=1, weights=rho[:, 1:])).reshape(analysis['Median'].shape)
        analysis['speed'] = analysis['Mean'][1:] - analysis['Mean'][:-1]
        analysis['acc'] = analysis['speed'][1:] - analysis['speed'][:-1]

        return analysis
    
def plot_analysis(analysis_rep, L, times, nums):
    fig, ax = plt.subplots(3, height_ratios=[3, 1, 1])
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
