from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy.sparse as sparse
import os

        
def defect_density(psi):
    L = len(psi)//3
    
    # Upper row
    edge_01 = np.arange(1,3*L,3)%(3*L)
    edge_10 = np.arange(3,3*(L+1),3)%(3*L) 
    edge_11 = np.arange(4,3*(L+1),3)%(3*L)
    
    # Lower row
    edge_02 = np.arange(2,3*L,3)%(3*L)
    edge_12 = np.arange(5,3*(L+1),3)%(3*L)
    
    psi_down = psi[edge_01] + psi[edge_10] + psi[edge_11] - 1
    psi_up = psi[edge_02] + psi[edge_10] + psi[edge_12] - 1
    
    return np.roll(psi_down + psi_up, 1)

def defect_density_point(psi):
    # L = len(psi)//3
    L = psi.shape[-1]//3
    
    # Upper row
    edge_01 = np.arange(1,3*L,3)%(3*L)
    edge_10 = np.arange(3,3*(L+1),3)%(3*L) 
    edge_11 = np.arange(4,3*(L+1),3)%(3*L)
    
    # Lower row
    edge_02 = np.arange(2,3*L,3)%(3*L)
    edge_12 = np.arange(5,3*(L+1),3)%(3*L)
    
    psi_down = (psi[:, edge_01] + psi[:, edge_10] + psi[:, edge_11] + 1) % 2
    psi_up = (psi[:, edge_02] + psi[:, edge_10] + psi[:, edge_12] + 1) % 2
    
    return np.roll(psi_down + psi_up, 1)

def defect_density_points_quantum(configs,psi):
    psi2 = np.abs(psi**2)
    density = defect_density_point(configs)
    charge = (density.T*psi2.T).T
    charge = np.sum(charge, axis=0)
    return charge
    

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

def get_initial_config_point(L, d):
    if (d < 1 or d >=L):
        raise ValueError("d= {} provided can't be to close to other defect".format(d))
    # print(L)
    # print(d)
    # print(configs.shape)
    c0 = np.zeros((1,3*L), dtype=np.int8)
    c0[0, 0] = 0
    c0[0, 2] = 0
    defect = int(d)
    for i in range(0,defect):
        c0[0, 3*i + 1 + i%2] = 1
    for i in range(defect + 1,L,2):
        c0[0, 3*i + 1] = 1
        c0[0, 3*i + 2] = 1
        
    c0[0, -1] = 0
    c0[0, -2] = 0
    if (L - d) % 2 == 0:
        c0[0, -3] = 1
    
    return c0

def get_initial_config_point_quantum(L,d, configs):
    dim = configs.shape[0]
    c0 = get_initial_config_point(L, d)
    i0 = np.where(np.dot(configs,c0.T)//np.sum(c0)==1)[0][0]
    psi = np.zeros((dim, 1)); 
    psi[i0] = 1.
    
    return psi

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
    # print(hops.shape)
    any_hops = np.argwhere(np.any(hops <= 2, axis=1))
    # print(np.argwhere(any_hops))
    h_hops = np.delete(hops, any_hops, axis=0)
    #h_hops = np.delete(h_hops, np.any(h_hops >= 3*L, axis=1), axis=0)
    # print(h_hops.shape)
    
    H_hops = list(map(hop, h_hops))
    return H_hops
    
@dataclass
class hop:
    sites : np.ndarray
    def __call__(self, config):
        return self.apply(config)
    def apply(self, config):
        if (config[self.sites[0]] != config[self.sites[3]]) and (config[self.sites[1]] + config[self.sites[2]] == 1) and (config[self.sites[4]] + config[self.sites[5]] == 1):  
            config[self.sites[[0,3]]] = 1 - config[self.sites[[0,3]]]
        return config

@dataclass
class Gate_ring:
    i : int
    def __call__(self, config):
       
        #print(type(config))
        cond = np.logical_and(np.logical_and(config[:, 3*self.i] == config[:, 3*(self.i + 1)],config[:, 3*self.i + 1] == config[:, 3*self.i + 2]), config[:, 3*self.i] != config[:, 3*self.i + 1])
        if np.any(cond):
            indices = range(3*self.i ,3*(self.i+1) + 2)
            start, end = indices[0], indices[-1]
            config[cond, start:end] = 1 - config[cond, start:end]
        # if (config[3*self.i] == config[3*(self.i + 1)]) and (config[3*self.i + 1] == config[3*self.i + 2]):
            # config[range(3*self.i ,3*(self.i+1) + 1)] = 1 - config[range(3*self.i ,3*(self.i+1) + 1)]
        return
    
@dataclass
class Gate_hop:
    # Verify this function! Is the return correct?
    i : int
    def __call__(self, config):
        #print(type(config))
        cond_top_right = np.logical_and(np.logical_and(config[:, 3*(self.i + 1)] == config[:, 3*(self.i + 1)+2],
                                                        config[:, 3*(self.i + 1)+2] == config[:, 3*self.i + 2]   ),
                                         config[:, 3*self.i + 2] == 0)
        
        cond_top_left = np.logical_and(np.logical_and(config[:, 3*(self.i - 1) + 2] == config[:, 3*self.i],
                                                      config[:, 3*self.i] == config[:, 3*self.i + 2]) ,
                                       config[:, 3*self.i + 2] == 0)
        
        cond_bottom_right = np.logical_and(np.logical_and(config[:, 3*self.i + 1] == config[:, 3*(self.i+1)] ,
                                                          config[:, 3*(self.i+1)] == config[:, 3*(self.i+1) + 1]),
                                           config[:, 3*(self.i+1) + 1] == 0)
        
        cond_bottom_left = np.logical_and(np.logical_and(config[:, 3*(self.i - 1) + 1] == config[:, 3*self.i], 
                                                         config[:, 3*self.i] == config[:, 3*self.i + 1]),
                                          config[:, 3*self.i + 1] == 0)
        # print(cond_top_right)
        if np.any(cond_top_right):
            config[cond_top_right, 3*self.i + 1], config[cond_top_right, 3*(self.i + 1)] = config[cond_top_right, 3*(self.i + 1)], config[cond_top_right, 3*self.i + 1]
        

        # print(cond_top_left)
        if np.any(cond_top_left):
            config[cond_top_left, 3*(self.i + 1)], config[cond_top_left,3*self.i + 2] = config[cond_top_left,3*self.i + 2], config[cond_top_left,3*(self.i + 1)]
        
        # print(cond_bottom_right)
        if np.any(cond_bottom_right):
            config[cond_bottom_right, 3*self.i + 2], config[cond_bottom_right, 3*(self.i + 1)] = config[cond_bottom_right,3*(self.i + 1)], config[cond_bottom_right,3*self.i + 2]
    
        # print(cond_bottom_left)
        if np.any(cond_bottom_left):
            config[cond_bottom_left, 3*self.i + 1], config[cond_bottom_left,3*(self.i + 1)] = config[cond_bottom_left, 3*(self.i + 1)], config[cond_bottom_left, 3*self.i + 1]
        
        # if config[3*(self.i + 1)] == config[3*(self.i + 1)+2] == config[3*self.i + 2] == 0:
        #     # Top-right
        #     config[3*self.i + 1], config[3*(self.i + 1)] = config[3*(self.i + 1)], config[3*self.i + 1]
        # elif config[3*(self.i - 1) + 2] == config[3*self.i] == config[3*self.i + 2] == 0:
        #     # Top-left
        #     config[3*(self.i + 1)], config[3*self.i + 2] = config[3*self.i + 2], config[3*(self.i + 1)]
        # elif config[3*self.i + 1] == config[3*(self.i+1)] == config[3*(self.i+1) + 1] == 0:
        #     # Bottom-right
        #     config[3*self.i + 2], config[3*(self.i + 1)] = config[3*(self.i + 1)], config[3*self.i + 2]
        # elif config[3*(self.i - 1) + 1] == config[3*self.i] == config[3*self.i + 1] == 0:
        #     # Bottom-left
        #     config[3*self.i + 1], config[3*(self.i + 1)] = config[3*(self.i + 1)], config[3*self.i + 1]
        return
            
            
def plot_conf(c):
    c=c.reshape(c.size)
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
            
        plt.plot([i,i],[0,1],color[int(c[3*i])],linewidth=3*c[3*i]+1) #vertical
        plt.plot([i,i+1],[0,0],color[int(c[3*i+1])],linewidth=3*c[3*i+1]+1) # horizontal
        plt.plot([i,i+1],[1,1],color[int(c[3*i+2])],linewidth=3*c[3*i+2]+1) # vertical
        
    plt.axis('off')
    stripped = str(c).translate(str.maketrans({"[": "", "]": "", " ": "", "\n":""}))
    # print([stripped[i:i + 3] for i in range(0, len(stripped), 3)])
    
def promote_psi_classical(psi, H_ring, H_hop, prob):
    L = psi.shape[-1]//3
    rng = np.random.default_rng()
    shift = rng.choice([0,1,2], 1)
    indices = np.arange(1+shift%3, L-2, 3)
    gates_i = rng.choice([True, False], size=(psi.shape[0], len(indices)), p =[prob, 1 - prob])

    apply = np.empty(gates_i.shape, dtype=object)
    rings_i = np.argwhere(gates_i)
    hops_i = np.argwhere(np.logical_not(gates_i))
    apply[rings_i[:,0], rings_i[:,1]] = H_ring[indices[rings_i[:,1]]]
    apply[hops_i[:,0], hops_i[:,1]] = H_hop[indices[hops_i[:,1]]]

    def apply_gate(f):
        return f[0](f[1])

    for row_gate in apply.T:
        zips = np.array(list(zip(row_gate, psi)), dtype=object)
        np.apply_along_axis(apply_gate, 1, zips)

    
def check_detailed_balance(L, times, d, prob=0.5, interval=10):
    from IPython import display
    
    size = 1
    H_ring = np.array([Gate_ring(i) for i in range(1,L - 1)], dtype=object)
    H_hop = np.array([Gate_hop(i) for i in range(1, L - 1)], dtype=object)
    psi = np.repeat(get_initial_config_point(L, d), size, axis=0).reshape(size, 1, 3*L)
    
    states = {psi.tobytes(): 1}
    state_vars = [0]
    for i in range(2, times):
        promote_psi_classical(psi, H_ring, H_hop, prob)  
            
        if psi.tobytes() in states:
            states[psi.tobytes()] += 1
        else:
            states[psi.tobytes()] = 1
        var_i =  np.var(list(states.values()))
        state_vars.append(np.sqrt(var_i)/i)
        if i % interval == 0:
            display.clear_output(wait=True)
            plt.plot(state_vars)
            plt.annotate(str(state_vars[-1]), (i,state_vars[-1]))
            plt.show()
        
        
###################
# Old code
    
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
        
def load_data(L):
    configs = load_configs('matrices/basis_L{}.dat'.format(L))
    H_ring = load_matrix('matrices/matrix_ring_L{}.dat'.format(L))
    H_hopp = load_matrix('matrices/matrix_hopp_L{}.dat'.format(L))
    
    
    print("#######################")
    
    return {"H_ring" : H_ring, "H_hopp" : H_hopp, "configs" : configs}

def defect_density_old(configs,psi):
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