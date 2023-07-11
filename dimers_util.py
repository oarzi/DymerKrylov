from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy.sparse as sparse
import os
from numba import int32, boolean
from numba.experimental import jitclass
from numba import jit

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
        raise ValueError("d= {} provided can't be to close to other defect".format(d))

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
    c0 = get_initial_config_point(L, d)
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
        # plt.plot([L,L],[0,1],color[int(c[3*L])],linewidth=3*c[3*L]+1) #vertical

        plt.axis('off')
        # stripped = str(c).translate(str.maketrans({"[": "", "]": "", " ": "", "\n":""}))
        # print([stripped[i:i + 3] for i in range(0, len(stripped), 3)])

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

            hop_list_up2 = np.array([[0,1,0,0,0,0,0]])
            
            self.hop_list = [hop_list_up1, hop_list_up2]
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
        old = np.array(config[:, 3*self.i:3*(self.i+2) + (0 if self.max_i else 1)])
        idx_list = [np.where(( old == hop_conf[0,None,None]).all(axis=-1).any(0))[0] for hop_conf in self.hop_list] 
        
        for idx, hop_conf in zip(idx_list, self.hop_list):
            config[idx, 3*self.i:3*(self.i+2) + (0 if  self.max_i else 1)] = self.rng.choice(hop_conf, size=idx.size)
        
        return config
    
def test_charge(psi, i):
    charge = defect_density_point(config)
    charge_fail = np.argwhere(np.sum(charge, axis=1) != 2)
    if charge_fail.size > 0:
        print(i)
        print(charge_fail)
        plot_conf(config[charge_fail])
        print(charge[charge_fail])
        raise SystemExit("Charge is not conserved")


def promote_psi_classical(psi, H_ring, H_hop, prob_ring):
    rng = np.random.default_rng()
    shift = rng.choice([0, 1, 2], 1)
    indices = np.arange(shift + 0, psi.shape[1]//3-1, 3)
    rng.shuffle(indices)
    gates_i = rng.choice([True, False], size=(indices.size, psi.shape[0]), p =[prob_ring, 1 - prob_ring])
                
    for i, row_gate in zip(indices, gates_i):
        rings_i = np.nonzero(row_gate)
        psi[rings_i] =  H_ring[i](psi[rings_i])

        hops_i = np.nonzero(np.logical_not(row_gate))
        psi[hops_i] = H_hop[i](psi[hops_i])

    return psi

    
def check_detailed_balance(L, times, d, gate, prob_ring=0.5, interval=10, size=1,test_charge=False):
    from IPython import display
    
    H_ring = np.array([gate(i, False) for i in range(0, L - 1)], dtype=object)
    H_hop = np.array([gate(i, True, False if i < L -2 else True) for i in range(0, L - 1)], dtype=object)
    states = {state.tobytes() : 0 for state in load_configs('matrices/basis_L{}.dat'.format(L))}

    psi = get_initial_config_point(L, d, size)
    state_vars = []
    
    for i in range(1, times):
        for conf in psi:
                states[conf.tobytes()] += 1
        count = np.bincount(list(states.values()))
        state_vars.append(np.std(count)/(i*size))

        psi = promote_psi_classical(psi, H_ring, H_hop, prob_ring)

        rho = np.mean(defect_density_point(psi), axis=0)
        
        if i % interval == 0:
            display.clear_output(wait=True)
            plt.subplot(3, 1, 1)
            plt.plot(state_vars)
            plt.title("std_i/i*size")
            plt.annotate(str(state_vars[-1]), (i,state_vars[-1]))
            
            plt.subplot(3, 1, 2)
            charge = defect_density_point(psi)
            plt.plot(rho[1:])
            plt.title("Charge distribution")
            
            plt.subplot(3, 1, 3)
            plt.plot(count)
            plt.title("bin count")
            
            plt.tight_layout()
            plt.show()
    print(len(states))
    return

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
        print(" load_configs {} -File not found!".format(fn))