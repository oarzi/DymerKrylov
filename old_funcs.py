import pickle
from dimers_util import *
from multiprocessing import Pool, Queue, Process, Semaphore, Manager
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
from itertools import repeat
from str import ljust


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