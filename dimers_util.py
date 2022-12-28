from dataclasses import dataclass
import numpy as np
from numpy.random import SeedSequence
ss = SeedSequence(12345)
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy.sparse as sparse


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

def get_h_ring(L):
    i = np.arange(1,L - 1)
    hrings = np.stack([3*i, 3*((i + 1) %L), 3*i + 1, 3*i + 2]).T
    
    H_ring = list(map(ring, hrings))
    return H_ring

@dataclass
class ring:
    sites : np.ndarray
    def __call__(self, config):
        return self.apply(config)
    def apply(self, config):
        if (config[self.sites[0]] == config[self.sites[1]]) and (config[self.sites[2]] == config[self.sites[3]]) and (config[self.sites[0]] != config[self.sites[2]]):  
            config[self.sites] = 1- config[self.sites]
        return config


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
    print(hops.shape)
    h_hops = np.delete(hops, np.any(hops <= 2, axis=1), axis=0)
    h_hops = np.delete(h_hops, np.any(h_hops >= 3*L, axis=1), axis=0)
     
    print(h_hops.shape)
    H_hops = list(map(hop, h_hops))
    return H_hops
    
@dataclass
class hop:
    sites : np.ndarray
    def __call__(self, config):
        return self.apply(config)
    def apply(self, config):
        if (config[self.sites[0]] != config[self.sites[3]]) and (config[self.sites[1]] + config[self.sites[2]]) % 2 and (config[self.sites[4]] + config[self.sites[5]]):  
            config[self.sites[[0,3]]] = 1 - config[self.sites[[0,3]]]
        return config