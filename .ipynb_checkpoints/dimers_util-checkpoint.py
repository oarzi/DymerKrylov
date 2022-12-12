from dataclasses import dataclass
import numpy as np
from numpy.random import SeedSequence
ss = SeedSequence(12345)
import matplotlib.pyplot as plt
import numpy as np
import struct
import sys
import os
import scipy.sparse as sparse
import time
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm
import re
import pickle
from itertools import repeat
from multiprocessing import Pool
import time

def charge_density(configs,psi):
    return np.sum(configs.T*np.abs(psi)**2,axis=1)
        
def defect_density(configs,psi):
    L = configs.shape[1]//3
    
    edge_01 = np.arange(1,3*L,3)%(3*L)
    edge_10 = np.arange(3,3*(L+1),3)%(3*L) 
    edge_11 = np.arange(4,3*(L+1),3)%(3*L)
    
    edge_02 = np.arange(2,3*L,3)%(3*L)
    edge_12 = np.arange(5,3*(L+1),3)%(3*L)
    
    d  = np.multiply((configs[:,edge_01] + configs[:,edge_10] + configs[:,edge_11] - 1),np.abs(psi)**2)
    d += np.multiply((configs[:,edge_02] + configs[:,edge_10] + configs[:,edge_12] - 1),np.abs(psi)**2)
    
    d = d.sum(axis=0)
            
    return np.roll(d,1)

def defect_density(psi):
    L = len(psi)//3
    edge_01 = np.arange(1,3*L,3)%(3*L)
    edge_10 = np.arange(3,3*(L+1),3)%(3*L) 
    edge_11 = np.arange(4,3*(L+1),3)%(3*L)
    
    edge_02 = np.arange(2,3*L,3)%(3*L)
    edge_12 = np.arange(5,3*(L+1),3)%(3*L)
    
    psi1 = psi[edge_01] + psi[edge_10] + psi[edge_11] - 1
    psi2 = psi[edge_02] + psi[edge_10] + psi[edge_12] - 1
    return np.roll(psi1 + psi2, 1)

def plot_conf(c):
    L = len(c)//3
    color = ['k','r']
    plt.figure(figsize=[L,1])
    for i in range(L):
        plt.plot([i,i],[0,1],color[int(c[3*i])],linewidth=3*c[3*i]+1) #vertical
        plt.plot([i,i+1],[0,0],color[int(c[3*i+1])],linewidth=3*c[3*i+1]+1) # horizontal
        plt.plot([i,i+1],[1,1],color[int(c[3*i+2])],linewidth=3*c[3*i+2]+1) # vertical
    plt.axis('off')
    stripped = str(c).translate(str.maketrans({"[": "", "]": "", " ": "", "\n":""}))
    # print([stripped[i:i + 3] for i in range(0, len(stripped), 3)])
    
def load_matrix(fn):
    if os.path.isfile(fn)==True and os.path.getsize(fn)>0:
        fin = open(fn,'rb')
        dim = int(struct.unpack('i', fin.read(4))[0])
        nnz = int(struct.unpack('i', fin.read(4))[0])
        print (dim,nnz)

        a=np.array(np.fromfile(fin, dtype=np.int32))
        fin.close()
        a=np.reshape(a,(nnz,3))
        H=sparse.csr_matrix( (a[:,2],(a[:,0],a[:,1])), shape=(dim,dim),dtype=np.float64)
        return H
    else:
        print("File not found!")

def load_configs(fn):
    if os.path.isfile(fn)==True and os.path.getsize(fn)>0:
        fin = open(fn,'rb')
        dim = int(struct.unpack('i', fin.read(4))[0])
        L = int(struct.unpack('i', fin.read(4))[0])
        print (dim,3*L)

        a=np.array(np.fromfile(fin, dtype=np.int8))
        fin.close()
        return np.reshape(a,(dim,3*L))
    else:
        print("File not found!")
        
def load_data(L):
    H_ring = load_matrix('matrices/matrix_ring_L{}.dat'.format(L))
    H_hopp = load_matrix('matrices/matrix_hopp_L{}.dat'.format(L))
    configs = load_configs('matrices/basis_L{}.dat'.format(L))
    
    print("#######################")
    
    return {"H_ring" : H_ring, "H_hopp" : H_hopp, "configs" : configs}

def get_initial_state(L, configs, dim, d=0):
    # print(L)
    # print(d)
    # print(configs.shape)
    c0 = np.zeros(3*L, dtype=np.int8)
    c0[0] = 1
    c0[2] = 1
    defect = int(L//2+d)
    for i in range(1,defect):
        c0[3*i + 1 + (i+1)%2] = 1
    for i in range(defect,L):
        c0[3*i] = 1
    i0 = np.where(np.dot(configs,c0)//np.sum(c0)==1)[0][0]
    psi = np.zeros((dim,1)); 
    psi[i0] = 1.
    
    return psi, i0

def get_initial_config(L, d):
    if (d < 2):
        raise ValueError("d= {} provided can't be to close to other defect".format(d))
    # print(L)
    # print(d)
    # print(configs.shape)
    c0 = np.zeros(3*L, dtype=np.int8)
    c0[0] = 1
    c0[2] = 1
    defect = int(d)
    for i in range(1,defect):
        c0[3*i + 1 + (i+1)%2] = 1
    for i in range(defect,L):
        c0[3*i] = 1
    
    return c0

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
    print(L, times, list(H.keys()), d, _p)
    H_ring, H_hopp, = H['H_ring'], H['H_hopp']
    gates = H_ring + H_hopp
    p_array = np.concatenate((np.ones(len(H_ring)), _p*np.ones(len(H_hopp))))/(len(H_ring)+_p*len(H_hopp))
    psi =  get_initial_config(L, d)
    rho = np.array([defect_density(psi)])
    
    for _ in  tqdm(range(times)):
        plot_conf(psi)
        gate = np.random.choice(gates, p=p_array)
        psi = gate.apply(psi)
        rho = np.row_stack((rho ,defect_density(psi)))
    plot_conf(psi)
    return rho

def classical_evolution(L, times, H, d=0, nums=1, steps=False, p=1):
    vc = np.vectorize(classical_evolutions_single2, otypes='O', cache=True)
    
    results = vc([L]*nums, [times]*nums, [H]*nums, [d]*nums, [p]*nums)
    
    rhos = np.array(results)
    rhos = np.sum(rhos, axis=0)/nums
    
    return (rhos, [res for res in results[1]]) if steps else rhos

def parallel_analysis(L, times, d, nums):
    H_ring, H_hopp = get_h_ring(L), get_h_hop(L)
    print(len(H_ring), len(H_hopp))
    H = {'H_ring' : H_ring, 'H_hopp' : H_hopp}
    with Pool(6) as p:
        c_rhos =  p.starmap(classical_evolution, [(L, times, H, d, nums) for d in d], chunksize=1)
        p.close()
        p.join()
    analysis_rhos =  [analyze(rho) for rho in c_rhos]
    with open('analysis{}_{}_{}.pickle'.format(L, times, time.strftime("%Y_%m_%d_%H_%M")), 'wb') as handle:
        pickle.dump(analysis_rhos, handle)
    return analysis_rhos


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
    
    
def analyze(rho):
    analysis = {}
    analysis['rho'] = rho
    analysis['Median'] = 1 + np.sum((np.cumsum(rho[:,1:],axis=1)<0.5).astype(int),axis=1).reshape(rho.shape[0])
    sites = [np.arange(1, rho.shape[1])]
    
    
    analysis['Mean'] = np.average(np.repeat(sites,rho.shape[0],axis=0), axis=1, weights=rho[:, 1:]).reshape(analysis['Median'].shape)
    analysis['std'] = np.sqrt(np.average((np.repeat(sites,rho.shape[0],axis=0) - analysis['Mean'].reshape(rho.shape[0],1))**2 , axis=1, weights=rho[:, 1:])).reshape(analysis['Median'].shape)
    analysis['speed'] = analysis['Mean'][1:] - analysis['Mean'][:-1]
    analysis['acc'] = analysis['speed'][1:] - analysis['speed'][:-1]
    

    return analysis

# def ring(sites):
#     def apply(config):
#         if (config[sites[0]] == config[sites[1]]) and (config[sites[2]] == config[sites[3]]) and (config[sites[0]] != config[sites[2]]):  
#             config = 1- config
#         return config
#     return apply

# def hop(sites):
#     def apply(config):
#         if (config[sites[0]] != config[sites[3]]) and (config[sites[1]] + config[sites[2]]) % 2 and (config[sites[4]] + config[sites[5]]):  
#             config[sites[[0,3]]] = 1 - config[sites[[0,3]]]
#         return config
#     return apply

@dataclass
class ring:
    sites : np.ndarray
    def apply(self, config):
        if (config[self.sites[0]] == config[self.sites[1]]) and (config[self.sites[2]] == config[self.sites[3]]) and (config[self.sites[0]] != config[self.sites[2]]):  
            config[self.sites] = 1- config[self.sites]
        return config

@dataclass
class hop:
    sites : np.ndarray
    def apply(self, config):
        if (config[self.sites[0]] != config[self.sites[3]]) and (config[self.sites[1]] + config[self.sites[2]]) % 2 and (config[self.sites[4]] + config[self.sites[5]]):  
            config[self.sites[[0,3]]] = 1 - config[self.sites[[0,3]]]
        return config
    

def get_h_ring(L):
    i = np.arange(1,L - 1)
    hrings = np.stack([3*i, 3*((i + 1) %L), 3*i + 1, 3*i + 2]).T
    
    H_ring = list(map(ring, hrings))
    return H_ring

def get_h_hop(L):
    i = np.arange(0,L);
    
    hop1 = np.stack([3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 2,
                     3 * ((i + 2) % L) + 1, 3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 1]).T
    
    hop2 = np.stack([3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 2,
                     3 * ((i + 3) % L) + 1, 3 * ((i + 4) % L) + 0, 3 * ((i + 4) % L) + 1]).T

    hop3 = np.stack([3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 1, 3 * ((i + 2) % L) + 1,
                     3 * ((i + 1) % L) + 2, 3 * ((i + 1) % L) + 0, 3 * (i % L) + 2]).T

    hop4 = np.stack([3 * ((i + 1) % L) + 0, 3 * (i % L) + 1,       3 * ((i + 1) % L) + 1,
                     3 * ((i + 1) % L) + 2, 3 * ((i + 2) % L) + 0, 3 * ((i + 2) % L) + 2]).T

    hop5 = np.stack([3 * ((i + 2) % L) + 1, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 1,
                     3 * ((i + 1) % L) + 1, 3 * ((i + 0) % L) + 1, 3 * ((i + 1) % L)]).T

    hop6 = np.stack([3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 2,
                     3 * ((i + 1) % L) + 2, 3 * ((i + 0) % L) + 2, 3 * ((i + 1) % L)]).T
    
    hops = np.vstack((hop1, hop2, hop3, hop4, hop5, hop6))
    h_hops = np.delete(hops, np.any(hops <= 2, axis=1), axis=0)
     
    H_hops = list(map(hop, h_hops))
    return H_hops
    
    