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
    # print(hops.shape)
    any_hops = np.argwhere(np.any(hops <= 2, axis=1))
    # print(np.argwhere(any_hops))
    h_hops = np.delete(hops, any_hops, axis=0)
    #h_hops = np.delete(h_hops, np.any(h_hops >= 3*L, axis=1), axis=0)
    # print(h_hops.shape)
    
    H_hops = list(map(hop, h_hops))
    return H_hops


@dataclass
class ring:
    sites : np.ndarray
    def __call__(self, config):
        return self.apply(config)
    def apply(self, config):
        if (config[self.sites[0]] == config[self.sites[1]]) and (config[self.sites[2]] == config[self.sites[3]]) and (config[self.sites[0]] != config[self.sites[2]]):  
            config[self.sites] = 1- config[self.sites]
        return config



@dataclass
class hop:
    sites : np.ndarray
    def __call__(self, config):
        return self.apply(config)
    def apply(self, config):
        if (config[self.sites[0]] != config[self.sites[3]]) and (config[self.sites[1]] + config[self.sites[2]] == 1) and (config[self.sites[4]] + config[self.sites[5]] == 1):  
            config[self.sites[[0,3]]] = 1 - config[self.sites[[0,3]]]
        return config
    
@jit(nopython=True)
def jit_ring(i, config, rng):
    rng = np.random.default_rng()
    idx, = np.where((config[:, 3*self.i:3*(self.i+1) + 1] == self.ring_list[:,None]).all(axis=-1).any(0))    
    config[idx, 3*self.i:3*(self.i+1) + 1] = self.rng.choice(self.ring_list, size=idx.size)

    return config


def get_initial_config(L, d, siz):
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
        
    psi = np.repeat(c0, size, axis=0)
    
    return c0

@dataclass
class Gate1:
    i: int
    do_hop: bool
    
    #def __post_init__(self):
     #   self.rng = np.random.default_rng()
    
    def __call__(self, config):
        #print(config.shape)
        #which = self.rng.choice([True, False], p =[self.prob, 1 - self.prob])
        return self.hop(config) if self.do_hop else self.ring(config)
    
    def ring(self, config):
        cond = np.logical_and(np.logical_and(config[:, 3*self.i] == config[:, 3*(self.i + 1)],config[:, 3*self.i + 1] == config[:, 3*self.i + 2]), config[:, 3*self.i] != config[:, 3*self.i + 1])
        if np.any(cond):
         #   print(cond)
            start, end = 3*self.i, 3*(self.i+1) + 1
            config[cond, start:end] = 1 - config[cond, start:end]

        return config
    
    def hop(self, config):
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

        if np.any(cond_top_right):
          #  print("tr", cond_top_right)
           # print(config[cond_top_right])
            config[cond_top_right, 3*self.i + 1], config[cond_top_right, 3*(self.i + 1)] = config[cond_top_right, 3*(self.i + 1)], config[cond_top_right, 3*self.i + 1]
            #print(config[cond_top_right])
        
        if np.any(cond_top_left):
            #print("tl", cond_top_left)
            #print(config[cond_top_left])
            config[cond_top_left, 3*(self.i + 1)], config[cond_top_left,3*self.i + 2] = config[cond_top_left,3*self.i + 2], config[cond_top_left,3*(self.i + 1)]
            #print(config[cond_top_left])
        
        if np.any(cond_bottom_right):
            #print("br", cond_bottom_right)
            #print(config[cond_bottom_right])
            config[cond_bottom_right, 3*self.i + 2], config[cond_bottom_right, 3*(self.i + 1)] = config[cond_bottom_right,3*(self.i + 1)], config[cond_bottom_right,3*self.i + 2]
            #print(config[cond_bottom_right])
    
        if np.any(cond_bottom_left):
            #print("bl", cond_bottom_left)
            #print(config[cond_bottom_left])
            config[cond_bottom_left, 3*self.i + 1], config[cond_bottom_left,3*(self.i + 1)] = config[cond_bottom_left, 3*(self.i + 1)], config[cond_bottom_left, 3*self.i + 1]
            #print(config[cond_bottom_left])
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