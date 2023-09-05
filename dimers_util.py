from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import struct
import scipy.sparse as sparse
import os
from numba import jit
import sys
from tqdm import tqdm
from scipy.constants import golden

@jit(nopython=True)
def defect_density_point(psi):
    # L = len(psi)//3
    L = psi.shape[-1]//3
    
    # Upper row
    edge_01 = np.arange(1, 3*(L - 1), 3)
    edge_10 = np.arange(3, 3*L, 3)
    edge_11 = np.arange(4, 3*L, 3)
    
    # Lower row
    edge_02 = np.arange(2,3*L - 1,3)
    edge_12 = np.arange(5,3*L,3)
    
    psi_down = (psi[:, edge_01] + psi[:, edge_10] + psi[:, edge_11] + 1) % 2
    psi_up = (psi[:, edge_02] + psi[:, edge_10] + psi[:, edge_12] + 1) % 2
    
    psi_down0 = (psi[:, 0] + psi[:, 1] + 1) % 2
    psi_up0 = (psi[:, 0] + psi[:, 2]  + 1) % 2
    psi_0 = (psi_down0 + psi_up0).reshape(psi.shape[0], 1)
    
    return np.hstack((psi_0 , psi_down + psi_up))

def defect_density_points_quantum(configs,psi):
    psi2 = np.abs(psi**2)
    density = defect_density_point(configs)
    charge = (density.T*psi2.T).T
    charge = np.sum(charge, axis=0)
    return charge
    

def get_initial_config_point(L, defect, size):
    if (defect < 1 or defect >=L):
        raise ValueError("d= {} provided can't be to close to other defect".format(defect))

    c0 = np.zeros((1,3*L), dtype=np.int8)
    c0[0, 0] = 0
    c0[0, 2] = 0

    for i in range(0,defect):
        c0[0, 3*i + 1 + i%2] = 1
   
    for i in range(defect + 1,L - 1,2):
        c0[0, 3*i + 1] = 1
        c0[0, 3*i + 2] = 1
        
    if (L - defect) % 2 == 0:
        c0[0, -3] = 1
        
    psi = np.repeat(c0, size, axis=0)
    
    return psi

def get_initial_config_point_quantum(L,d, configs):
    dim = configs.shape[0]
    c0 = get_initial_config_point(L, d, 1)
    i0 = np.where(np.dot(configs,c0.T)//np.sum(c0)==1)[0][0]
    psi = np.zeros((dim, 1)); 
    psi[i0] = 1.
    
    return psi

def plot_conf(psi):

    for c in psi:
        c=c.reshape(c.size).astype(np.int32)
        L = len(c)//3
        color = ['k','r']
        plt.figure(figsize=[L,1])
        if c[0] == c[2]  == 0:
                plt.scatter([0],[1],c='r', marker='+', s=10, linewidths=5) # defect
        for i in range(L):
            if c[3*i] + c[3*((i-1)%L)+2] + c[3*i+2] == 0:
                plt.scatter([i],[1],c='r', marker='+', s=10, linewidths=5) # defect
            if c[3*i] + c[3*((i-1)%L)+1] + c[3*i+1] == 0:
                plt.scatter([i],[0],c='r', marker='+', s=10, linewidths=5) # defect

            plt.plot([i,i],[0,1],color[c[3*i]],linewidth=3*c[3*i]+1) #vertical
            plt.plot([i,i+1],[0,0],color[c[3*i+1]],linewidth=3*c[3*i+1]+1) # horizontal
            plt.plot([i,i+1],[1,1],color[c[3*i+2]],linewidth=3*c[3*i+2]+1) # horizontal


        plt.axis('off')
    return

@dataclass
class Gate2:
    i: int
    do_hop: bool
    max_i : bool = False
    
    def __post_init__(self):
        self.rng = np.random.default_rng()
        self.ring_list = np.array([[1,0,0,1],
                                   [0,1,1,0]])
        if self.i == 0:
            hop_list_up1 = np.array([[0,1,0,0,0,0,1], 
                                     [0,1,0,0,0,1,0]])

            # hop_list_up2 = np.array([[0,1,0,0,0,0,0]])
            
            self.hop_list = [hop_list_up1]
        elif self.max_i:
            hop_list_up1 = np.array([[0,1,0,0,0,0], 
                                     [0,0,0,1,0,0]])

            hop_list_down1 = np.array([[0,0,1,0,0,0], 
                                       [0,0,0,1,0,0]])

            self.hop_list = [hop_list_up1, hop_list_down1]
        else:
            hop_list_up1 = np.array([[0,1,0,0,0,0,1], 
                                     [0,0,0,1,0,0,1],
                                     [0,1,0,0,0,1,0]])

            hop_list_up2 = np.array([[0,1,0,0,0,0,0], 
                                     [0,0,0,1,0,0,0]])

            hop_list_down1 = np.array([[0,0,1,0,0,0,1], 
                                       [0,0,1,0,1,0,0],
                                       [0,0,0,1,0,0,1]])

            hop_list_down2 = np.array([[0,0,1,0,0,0,0], 
                                       [0,0,0,1,0,0,0]])
            
            self.hop_list = [hop_list_up1, hop_list_up2, hop_list_down1, hop_list_down2]
        
    def __call__(self, config):
        return self.hop(config) if self.do_hop else self.ring(config)
    
    def ring(self, config):
        idx, = np.where((config[:, 3*self.i:3*(self.i+1) + 1] == self.ring_list[:,None]).all(axis=-1).any(0))    
        config[idx, 3*self.i:3*(self.i+1) + 1] = self.rng.choice(self.ring_list, size=idx.size)
        return config
    
    def hop(self, config):

        idx_list = [np.where(( config[:, 3*self.i:3*(self.i+2)+1] == hop_conf[0,None,None]).all(axis=-1).any(0))[0] for hop_conf in self.hop_list] 
        
        # moves = np.zeros((config.shape[0], 1))
        for idx, hop_conf in zip(idx_list, self.hop_list):
            # if idx:
                # flips = self.rng.choice(np.arange(hop_conf.shape[0]), size=idx.size)
                # config[idx, 3*self.i:3*(self.i+2) + (0 if  self.max_i else 1)] = hop_conf[flips]
                # flips[flips==2] = -1
                # moves[idx] = flips
            config[idx, 3*self.i:3*(self.i+2) + 1] = self.rng.choice(hop_conf, size=idx.size, p=[1/3, 1/3, 1/3] if hop_conf.shape[0] == 3 else [2/3, 1/3])

        
        return config
    
def test_charge(psi, i):
    charge = defect_density_point(psi)
    charge_fail = np.argwhere(np.sum(charge, axis=1) != 2)
    if charge_fail.size > 0:
        print(i)
        print(charge_fail)
        plot_conf(psi[charge_fail])
        print(charge[charge_fail])
        raise SystemExit("Charge is not conserved")


def promote_psi_classical(psi, H_ring, H_hop, prob_hop, wait = 5):
    for j in range(wait):
        rng = np.random.default_rng()
        shift = rng.choice([0, 1, 2], 1)
        indices = np.arange(shift + 0, psi.shape[1]//3-1, 3)
        rng.shuffle(indices)
        gates_i = rng.choice([False, True], size=(indices.size, psi.shape[0]), p =[1 - prob_hop, prob_hop])
                    
        for i, row_gate in zip(indices, gates_i):
            rings_i = np.nonzero(np.logical_not(row_gate))
            psi[rings_i] =  H_ring[i](psi[rings_i])
    
            hops_i = np.nonzero(row_gate)
            psi[hops_i] = H_hop[i](psi[hops_i])
        
    return psi

def promote_psi_classical2(psi, H_hop, prob_hop, wait=10):
    for j in range(wait):
        rng = np.random.default_rng()
        charge = defect_density_point(psi)
        where_hop = np.nonzero(charge[:,1:])[1]
        gates_j = rng.choice([False, True], size= psi.shape[0], p =[1 - prob_hop, prob_hop])
        
        for idx, gate in enumerate(gates_j):
            if gate:
                psi[idx] = H_hop[where_hop[idx]](psi[idx:idx+1])
            elif where_hop[idx] + 2 < psi.shape[1]//3 -1:
                psi[idx] = update_worm(psi[idx], where_hop[idx], np.random.default_rng())
        
    return psi

    
def check_detailed_balance(L, times, d, gate, prob_hop=0.5, interval=10, size=1,test_charge=False, save=False):
    from IPython import display
    
    H_ring = np.array([gate(i, False) for i in range(0, L - 1)], dtype=object)
    H_hop = np.array([gate(i, True, False if i < L -2 else True) for i in range(0, L - 1)], dtype=object)

    states = {state.tobytes() : 0 for state in load_configs(L)}
    
    psi = get_initial_config_point(L, d, size)
    psi = promote_psi_classical(psi, H_ring, H_hop, prob_hop, wait=max(250,L**2))
    
    state_vars = []
    N = len([*states])
    charge = defect_density_point(psi)
    rho = np.mean(charge, axis=0).reshape((1, psi.shape[1]//3))
    
    for i in tqdm(range(1, times+1)):
        
        psi = promote_psi_classical(psi, H_ring, H_hop, prob_hop, wait=interval)
        
        for conf in psi:
            states[conf.tobytes()] += 1
        
        charge = defect_density_point(psi)
        rho = np.vstack((rho, np.mean(charge, axis=0).reshape((1, psi.shape[1]//3))))
        count = np.bincount(list(states.values()))
        state_vars.append(np.std(list(states.values()))/(i*size))
            
        t_range = np.arange(1,1 + i)
        display.clear_output(wait=True)
        plt.figure(figsize=(16, 18))
        plt.subplot(3, 1, 1)
        plt.plot(t_range, state_vars, label=r'$\sigma$')
        plt.plot(t_range, np.sqrt(1/(N-1))*np.sqrt(1/(size*t_range)), label=r'$\sqrt{\frac{1}{N-1}}\frac{1}{\sqrt{S \times T}}$')
        plt.title("std_i/i*size")
        plt.yscale("log", base=10)
        plt.xscale("log", base=10)
        plt.tick_params(width=3, length=6, which ='both')
        plt.annotate(str(state_vars[-1]), (i,state_vars[-1]), fontsize=12)
        plt.legend(fontsize=20)
        
        plt.subplot(3, 1, 2)
        nz = np.nonzero(count)[0]
        plt.bar(nz/(i*size), count[nz], width=1/(i*size))
        plt.vlines(1/N, 0, np.max(count), colors='r', label=r'$\frac{1}{N}$')
        plt.title("bin count")
        plt.legend(fontsize=20)
        
        plt.subplot(3, 1, 3)
        mean_rho = np.mean(rho[1:, 1:], axis=0)
        plt.plot(np.arange(1,L), mean_rho, label="Sim")  
        ticks=[0.1, 0.3, 0.5]
        labels=[r'$10^{-1}$', r'$3 \times 10^{-1}$', r'$5 \times 10^{-1}$']
        minlog = int(np.floor(np.log10(min(mean_rho[np.nonzero(mean_rho)[0]]))))
        logticks = np.logspace(-2, minlog, -2-minlog+1)
        ticks = ticks + [*logticks]
        labels = labels + [r'$10^{}$'.format(i) for i in np.log10(logticks).astype(dtype=np.int32)]
        plt.yticks(ticks=ticks,
                   labels=labels)
        plt.plot(np.arange(1,L), np.log(golden)*np.e**(-np.log(golden)*np.arange(0,L-1)), label=r"$\log \phi e^{- \log \phi \times x}$")
        plt.yscale("log", base=10)
        plt.hlines(np.log(golden), 1, L-1, color='r')
        plt.title("Charge distribution")
        plt.legend(fontsize=20)
        
        plt.tight_layout()
        if save:
            plt.savefig("det_bal/detbal_L{}_d{}_size{}_p{}.png".format(L, d, size, prob_hop), format='png')
            plt.close()
        plt.show()
    print(len(states))
    
    
    return states, state_vars, mean_rho

def check_detailed_balance2(L, times, d, gate, prob_hop=0.5, interval=10, size=1,test_charge=False, save=False):
    from IPython import display
    
    H_ring = np.array([gate(i, False) for i in range(0, L - 1)], dtype=object)
    H_hop = np.array([gate(i, True, False if i < L -2 else True) for i in range(0, L - 1)], dtype=object)

    states = {state.tobytes() : 0 for state in load_configs(L)}
    psi = get_initial_config_point(L, d, size)

    state_vars = []
    
    rng = np.random.default_rng()
    for conf in psi:
        states[conf.tobytes()] += 1
    charge = defect_density_point(psi)
    rho = np.mean(charge, axis=0).reshape((1, psi.shape[1]//3))
        
    for i in tqdm(range(1, times)):

        gates_i = rng.choice([False, True], size= psi.shape[0], p =[1 - prob_hop, prob_hop])

        where_hop = np.nonzero(charge[:,1:])[1]
        
        for idx, gate in enumerate(gates_i):
            if gate:
                psi[idx] = H_hop[where_hop[idx]](psi[idx:idx+1])
            else:
                shift = rng.choice([1, 2], 1)
                indices = np.arange(where_hop[idx] + shift + 1, psi.shape[1]//3-1, 1)
                rings = rng.choice([False, True], size= indices.size)
                for ring_i in rng.permuted(indices[rings]):
                    psi[idx] = H_ring[ring_i](psi[idx:idx+1])
            states[psi[idx].tobytes()] += 1
        
        charge = defect_density_point(psi)
        rho = np.vstack((rho, np.mean(charge, axis=0).reshape((1, psi.shape[1]//3))))
                    
                # rings = where_hop[idx] + 1 + np.nonzero([(psi[idx, 3*i:3*(i+1) + 1] == ring_list[:,None]).all(axis=-1).any()  for i in range(where_hop[idx] + 1, psi.shape[1]//3-1)])[0]
                # if rings.size:
                #     psi[idx] = H_ring[rng.choice(rings)](psi[idx:idx+1])                             
            
        state_vars.append(np.std(list(states.values()))/(i*size))
        count = np.bincount(list(states.values()))
        if i % interval == 0:
            t_range = np.arange(0,i) 
            
            display.clear_output(wait=True)
            plt.figure(figsize=(15, 8))
            plt.subplot(3, 1, 1)
            plt.plot(t_range, state_vars)
            plt.title("std_i/i*size")
            plt.yscale("log", base=10)
            plt.xscale("log", base=10)
            plt.annotate(str(state_vars[-1]), (i,state_vars[-1]))
            
            plt.subplot(3, 1, 2)
            plt.plot(np.linspace(0,np.max(list(states.values()))/(i*size), len(count)), count)
            plt.vlines(1/len([*states]), 0, np.max(count), colors='r', label=r'$\frac{1}{N}$')
            # plt.vlines(np.mean(list(states.values())), 0, np.max(count), colors='r')
            plt.title("bin count")
            plt.legend()
            
            plt.subplot(3, 1, 3)
            if i < 750:
                plt.plot(np.arange(1,L), rho[-1, 1:], label="Sim")
            else:
                plt.plot(np.arange(1,L), np.mean(rho[750:, 1:], axis=0), label="Sim")   
            plt.plot(np.arange(1,L), np.log(golden)*np.e**(-np.log(golden)*np.arange(0,L-1)), label=r"$\log \phi e^{- \log \phi \times x}$")
            plt.yscale("log", base=10)
            plt.title("Charge distribution")
            plt.legend()
            
            plt.tight_layout()
            if save:
                plt.savefig("det_bal/detbal2_L{}_d{}_size{}_p{}.png".format(L, d, size, prob_hop), format='png')
                plt.close()
            plt.show()
    print(len(states))
    

    return states, state_vars


def check_detailed_balance3(L, times, d, gate, prob_hop=0.5, interval=10, size=1,test_charge=False, save=False):
    from IPython import display

    H_hop = np.array([gate(i, True, False if i < L -2 else True) for i in range(0, L - 1)], dtype=object)
    
    psi = get_initial_config_point(L, d, size)
    psi = promote_psi_classical2(psi,  H_hop, prob_hop, wait=max(250,L**2))

    states = {state.tobytes() : 0 for state in load_configs(L)}
    N = len([*states])
    state_vars = []
    charge = defect_density_point(psi)
    rho = np.mean(charge, axis=0).reshape((1, psi.shape[1]//3))

    for i in tqdm(range(1, times+1)):
        
        psi = promote_psi_classical2(psi,  H_hop, prob_hop, wait=interval)

        for conf in psi:
            states[conf.tobytes()] += 1
        
        charge = defect_density_point(psi)
        rho = np.vstack((rho, np.mean(charge, axis=0).reshape((1, psi.shape[1]//3))))
        state_vars.append(np.std(list(states.values()))/(i*size))
        count = np.bincount(list(states.values()))
        
        t_range = np.arange(1,1 + i)
        display.clear_output(wait=True)
        plt.figure(figsize=(16, 18))
        plt.subplot(3, 1, 1)
        plt.plot(t_range, state_vars, label=r'$\sigma$')
        plt.plot(t_range, np.sqrt(1/(N-1))*np.sqrt(1/(size*t_range)), label=r'$\sqrt{\frac{1}{N-1}}\frac{1}{\sqrt{S \times T}}$')
        plt.title("std_i/i*size")
        plt.yscale("log", base=10)
        plt.xscale("log", base=10)
        plt.tick_params(width=3, length=6, which ='both')
        plt.annotate(str(state_vars[-1]), (i,state_vars[-1]), fontsize=12)
        plt.legend(fontsize=20)
        
        plt.subplot(3, 1, 2)
        nz = np.nonzero(count)[0]
        plt.bar(nz/(i*size), count[nz], width=1/(i*size))
        plt.vlines(1/N, 0, np.max(count), colors='r', label=r'$\frac{1}{N}$')
        plt.title("bin count")
        plt.legend(fontsize=20)
        
        plt.subplot(3, 1, 3)
        mean_rho = np.mean(rho[1:, 1:], axis=0)
        plt.plot(np.arange(1,L), mean_rho, label="Sim")  
        ticks=[0.1, 0.3, 0.5]
        labels=[r'$10^{-1}$', r'$3 \times 10^{-1}$', r'$5 \times 10^{-1}$']
        minlog = int(np.floor(np.log10(mean_rho[-1])))
        logticks = np.logspace(-2, minlog, -2-minlog+1)
        ticks = ticks + [*logticks]
        labels = labels + [r'$10^{}$'.format(i) for i in np.log10(logticks).astype(dtype=np.int32)]
        plt.yticks(ticks=ticks,
                   labels=labels)
        plt.plot(np.arange(1,L), np.log(golden)*np.e**(-np.log(golden)*np.arange(0,L-1)), label=r"$\log \phi e^{- \log \phi \times x}$")
        plt.yscale("log", base=10)
        plt.hlines(np.log(golden), 1, L-1, color='r')
        plt.title("Charge distribution")
        plt.legend(fontsize=20)
        
        plt.tight_layout()
        if save:
            plt.savefig("det_bal/detbal3_L{}_d{}_size{}_p{}.png".format(L, d, size, prob_hop), format='png')
            plt.close()
        plt.show()
    print(len(states))
    
    return states, state_vars, mean_rho

@jit(nopython=True)
def update_worm(new_psi, where_hop, rng):
    row_curr = rng.integers(0, 2)
    if row_curr == 0:
        return new_psi
    row_curr = rng.integers(0, 2)
    # row_curr = 0 
    plaq_curr = rng.integers(where_hop + 2, new_psi.size//3)
    plaq0, row0 = plaq_curr, row_curr
    
    def get_dimer_idx(plaq, row, where_hop, psi_size):
        dimer_idx= [3*(plaq-1)+1, 3*plaq, 3*plaq+1] if row == 0 else [3*(plaq-1)+2, 3*plaq, 3*plaq+2]
        if plaq == where_hop +2:
            dimer_idx = dimer_idx[1:]
        elif plaq == psi_size//3 - 1:
            dimer_idx = dimer_idx[:-1]
        
        return dimer_idx
    
    def update_loc(plaq, row, idx):
        if idx > 3*plaq:     
            plaq = plaq+1
        elif idx < 3*plaq:
            plaq = plaq - 1
        else:
            row = (row+1)%2
        
        return plaq, row
    
    insert_idx = -1
    while True:
        # Remove dimer
        dimer_idx = get_dimer_idx(plaq_curr, row_curr, where_hop, new_psi.size)
        if insert_idx in dimer_idx:
            dimer_idx.remove(insert_idx)
        remove_dimer = np.array([new_psi[dix] for dix in dimer_idx]).nonzero()[0][0]
        remove_idx = dimer_idx[remove_dimer]
        new_psi[remove_idx] = 1 - new_psi[remove_idx]
        plaq_curr, row_curr = update_loc(plaq_curr, row_curr, remove_idx)
        
        # Add Dimer
        dimer_idx = get_dimer_idx(plaq_curr, row_curr, where_hop, new_psi.size)
        dimer_idx.remove(remove_idx)
        insert_idx = dimer_idx[rng.integers(0, len(dimer_idx))]
        new_psi[insert_idx] = 1 - new_psi[insert_idx]
        plaq_curr, row_curr = update_loc(plaq_curr, row_curr, insert_idx)

        if (plaq_curr == plaq0) and (row_curr == row0):
            break
    return new_psi

def load_configs(L):
    fn = "matrices/basis_L{}.dat".format(L)
    if os.path.isfile(fn)==True and os.path.getsize(fn)>0:
        fin = open(fn,'rb')
        dim = int(struct.unpack('i', fin.read(4))[0])
        L = int(struct.unpack('i', fin.read(4))[0])
        print (dim,3*L)

        a=np.array(np.fromfile(fin, dtype=np.int8))

        fin.close()
        configs = np.reshape(a,(dim,3*L))
        print(configs.shape)
        return configs
    else:
        print(" load_configs {} -File not found!".format(fn))
        
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
        print("load_matrix {} - File not found!".format(fn))
        
def load_data(L):
    H_ring = load_matrix('matrices/matrix_ring_L{}.dat'.format(L))
    H_hopp = load_matrix('matrices/matrix_hopp_L{}.dat'.format(L))
    
    
    print("#######################")
    
    return {"H_ring" : H_ring, "H_hopp" : H_hopp}
