from scipy.optimize import curve_fit
import pickle
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt
import time
import os
from dataclasses import dataclass, field
import argparse
import sys
import numpy as np
import subprocess
import lzma
from scipy.constants import golden

@dataclass
class Experiment: 
    results : list
    file_name : str = field(default="dimer_experiment")
    dir_name : str = field(default='/analyses')
    description : str = ''
    
    def save(self):
        with lzma.open(self.dir_name + "/" + self.file_name + ".pickle", 'wb', preset=9) as f:
            pickle.dump(self, f)
            print("Saved at {}".format(self.dir_name + "/" + self.file_name))
    
    @classmethod
    def load(cls, dir_path, description="" ):
        exp_files = []
        dir_paths = os.listdir(dir_path)
        for idx, path in enumerate(dir_paths):
            try:
                with lzma.open(dir_path + "/" +path, 'rb') as f:
                    print("Loading {}/{}: {} ...".format(idx + 1, len(dir_paths) ,path))
                    _e = pickle.load(f)
                    # print(type(_e))
                    if isinstance(_e, Experiment):
                        print("Experiment loaded")
                        exp_files.append(_e.results[0])
                    if isinstance(_e, Analysis):
                        print("Analysis loaded")
                        _e.analyze()
                        print(_e.p, _e.rho.shape)
                        exp_files.append(_e)
            except Exception as e:
                print(e)
                print("Failed: " + dir_path + "/" +path)

        experiment = Experiment(sorted(exp_files, key= lambda e: e.p),
                                "analyses/good", description=description)
        return experiment

@dataclass
class Analysis: 
    file_name : str
    dir_name : str
    L : int
    times: int
    d : int
    batch : int
    p : float
    rho : np.array
    analysis: dict = field(default_factory=dict, init=False)
    _rho : np.ndarray = field(init=False, repr=False)
    _psis : list = field(init=False, repr=False)
    psis : list = field(default_factory=list, init=False)
    
    def save(self):
        with lzma.open(self.dir_name + self.file_name + ".pickle", "wb", preset=9) as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with lzma.open(path, 'rb') as f:
            ana = pickle.load(f)
            ana.analyze()
            print(ana.p, ana.rho.shape)
            return ana
                
    @property
    def rho(self):
        return self._rho
    
    @rho.setter
    def rho(self, rho):
        self._rho = rho
        
    @property
    def psis(self):
        return self._psis
        
    @psis.setter
    def psis(self, psis):
        self._psis = []

    def analyze(self):
        # print("Analysis start")
        self.analysis = {}

        sites = np.arange(1, self.rho.shape[1]).reshape(1, self.rho.shape[1] - 1)
        weigths_avg = np.repeat(sites, self.rho.shape[0], axis=0)
        self.analysis['Mean'] = np.average(weigths_avg, axis=1,
                                           weights=self.rho[:, 1:])
        
        self.analysis['std'] = np.sqrt(np.average((np.repeat(sites, self.rho.shape[0], axis=0) -  
                                                   self.analysis['Mean'].reshape(self.rho.shape[0], 1))**2 , axis=1,
                                                  weights=self.rho[:, 1:])).reshape(self.rho.shape[0])

        # print("Analysis end")
        return self.analysis

def analyze_old(rho):
    # print("Analysis start")
    # print(self.rho)
    self.analysis = {}
    self.analysis['Median'] = 1 + np.sum((np.cumsum(self.rho[:,1:],axis=1)<0.5).astype(int),axis=1).reshape(self.rho.shape[0])

    sites = np.arange(1, self.rho.shape[1]).reshape(1, self.rho.shape[1] - 1)
    weigths_avg = np.repeat(sites, self.rho.shape[0], axis=0)
    self.analysis['Mean'] = np.average(weigths_avg, axis=1,
                                       weights=self.rho[:, 1:]).reshape(self.rho.shape[0])

    self.analysis['std'] = np.sqrt(np.average((np.repeat(sites, self.rho.shape[0], axis=0) -                        self.analysis['Mean'].reshape(self.rho.shape[0], 1))**2 , axis=1, weights=self.rho[:, 1:])).reshape(self.analysis['Median'].shape)

    self.analysis['speed'] = self.analysis['Mean'][1:] - self.analysis['Mean'][:-1]
    self.analysis['acc'] = self.analysis['speed'][1:] - self.analysis['speed'][:-1]
    # print("Analysis end")
    return self.analysis

def steady_state(results, times, x_min=1, x_max=-1):
    params_t = {}
    
    for ana in results:
        site_max = ana.rho.shape[1] if x_max == -1 else x_max
        x_range = np.arange(x_min, x_max)
        ana_times = ((ana.rho.shape[0] - 1)*times).astype(np.int32)
        p = [curve_fit(exponential_short, x_range , ana.rho[t,x_range], bounds=(0, 2), p0=0.1) for t in ana_times]
        # p = [dist_fit(ana.rho[:, 1:site_max], exponential, t, p0=1, bounds=(0, 3))[0] for t in ana_times]
        params_t[ana.p] = np.mean([k[0] for k in p])
        
    return params_t

def gaussian(t, a, b):
    return (1/(b*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((t-a)/b)**2)

def exponential_short(t, a):
    return a*np.exp(-a*(t - 1))

def exponential(t, a):
    return a*np.exp(-a*t)

def inv_pol(t, a):
    return a*np.exp(-a*t)

def dist_fit(rho, fit, t, p0=None):
    x_max, x_min = np.argwhere(rho[t] != 0)[-1][0], np.argwhere(rho[t] != 0)[0][0] - 5
    x_max, x_min = min([x_max, rho.shape[1]]), max([x_min, 1])
    popt, pcov = curve_fit(fit, np.arange(x_min, x_max), rho[t,x_min:x_max], bounds=(0, x_max),p0=p0)

    return popt, pcov, x_max, x_min

def plot_dist_scaled_p(ana_list, velocity, t, x_max, x_0, D, save=False, name=""):
    x_min = 1
    f, ax = plt.subplots(1, 1, figsize=(12,8))

    xrange = np.arange(1, x_max, dtype=np.int32)

    
    for ana, di, ti in zip(ana_list, D, t):
        ana_time = int(ana.rho.shape[0]*ti)
        scaled_x = (xrange-velocity[ana.p]*ana_time- x_0)/np.sqrt(di*ana_time)
        ax.plot(scaled_x, np.sqrt(di*ana_time)*ana.rho[ana_time, x_min : x_max], label='t={}, p={}'.format(ti, ana.p))

    ax.set_xlim(-10, 10)
    plt.title(r'$x \rightarrow \frac{x-vt}{\sqrt{t}}$' + ' for various p,t={}'.format(ana_list[0].rho.shape[0]), fontsize='large')
    plt.legend()
    # plt.show()
    if save:
        plt.savefig("figs/" + name + '.png', format='png')

def fit_scaled_dist(ana, velocity, t_fit, x_max, x0):
    xrange = np.arange(-x_max/2, x_max/2-1, dtype=np.int32)
    t_fit = int(ana.rho.shape[0]*t_fit)
    x_min = 1
    scaled_x = (xrange-velocity*t_fit - x0)/np.sqrt(t_fit)

    popt, pcov = curve_fit(gaussian, scaled_x, np.sqrt(t_fit)*ana.rho[t_fit, x_min : x_max], bounds = ([-5,0],[5,5]), p0=[1,1])
    return popt[1]


def fit_velocity(t, a, b):
    return  a*t +b
def extract_velocity(ana ,t_min, t_max):
    
    bound_low = [(min(ana.analysis['Mean'][t_min:t_max])-max(ana.analysis['Mean'][t_min:t_max]))/(np.argmin(ana.analysis['Mean'][t_min:t_max]) - np.argmax(ana.analysis['Mean'][t_min:t_max])), 0]
    bound_low[0] = -1 if np.isnan(bound_low[0]) else bound_low[0]
    bound_up = [0, ana.analysis['Mean'][t_min]+1]
    
    p0 = (bound_low[0]/2, ana.analysis['Mean'][t_min])
    
    popt, pcov = curve_fit(fit_velocity, np.arange(t_min, t_max), ana.analysis['Mean'][t_min:t_max],
                           bounds=(bound_low, bound_up), p0=p0)

    return popt, pcov


def plot_fit(ana, times,f, label, p0=None, log_scale_x=False, log_scale_y=False, site_max=-1):
    ana_times = ((ana.rho.shape[0] - 1)*times).astype(np.int32)
    plt.figure(1, figsize=(8,3))
    site_max = ana.rho.shape[0] if site_max == -1 else site_max
    res = {}
    for i, t in enumerate(zip(times, ana_times)):
        plt.subplot(100*len(times) + 10 +i+1)
        popt_t, pcov_t, x_max, x_min = dist_fit(ana.rho[:, 1:site_max], f, t[1], p0)
        print(x_min, x_max)
        res[t[0]]= (popt_t, pcov_t)
        print("Errors: {}".format(np.sqrt(np.diag(pcov_t))))
        xrange = np.arange(x_min, x_max)
        y = ana.rho[t[1],x_min:x_max]
        plt.plot(xrange, y, label="Simulation L={}".format(str(ana.L)))
        plt.plot(xrange, f(xrange, *popt_t),label="{} fit L={}, {}".format(label, str(ana.L), *popt_t))
        plt.title("p={}, L={}, d={}, t={}".format(ana.p, ana.L, ana.d, t[0]))
        if log_scale_x:
            plt.xscale("log", base=log_scale_x)
        if log_scale_y:
            plt.yscale("log", base=log_scale_y)
    if f == exponential:
        plt.plot(xrange, np.log(golden)*np.exp(-np.log(golden)*xrange), label=r"$\log \phi e^{- \log \phi \times x}$")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return res

def plot_analyses(analyses, label, save=False, title='', name='', log_scale_x=False, log_scale_y=False, t_max=-1):
    lwdt = 1

    fig, ax = plt.subplots(1, figsize=(13, 10))
        # fig, ax = plt.subplots(1, gridspec_kw={'height_ratios':[1, 1]}, figsize=(13, 10))
    if title:
        fig.suptitle(title)
    
    for a in analyses:
        a_label = "{}=".format(label) + str(a.__dict__[label])
        pos = a.analysis['Mean'][:t_max]
        ax.plot(pos, label=a_label, linewidth=lwdt)
        x = len(pos)//2
        y = pos[x]
        ax.annotate(a_label, (x,y))
    ax.legend()
    ax.set_title("Mean position")

    # for a in analyses:
    #     ax[1].plot(a.analysis['speed'][:t_max], label="{}=".format(label) + str(a.__dict__[label]), linewidth=lwdt)
    # ax[1].legend()
    # ax[1].set_title("Speed")

    # for a in analyses:
    #     ax[2].plot(a.analysis['acc'][:t_max], label="{}=".format(label) + str(a.analysis[label]), linewidth=lwdt)
    # ax[2].set_title("acceleration")
    
    fig.tight_layout()
    if log_scale_x:
        ax[0].set_xscale("log", base=log_scale_x)
        # ax[1].set_xscale("log", base=log_scale_x)
        # ax[2].set_xscale("log", base=log_scale_x)
    if log_scale_y:
        ax[0].set_yscale("log", base=log_scale_y)
        # ax[1].set_yscale("log", base=log_scale_y)
        # ax[2].set_yscale("log", base=log_scale_y)
    if save and name:
        plt.savefig("figs/" + name + '.png', format='png')
    plt.show()
    
def plot_analyses_old(analyses, label, save=False, title='', name=''):
    lwdt = 1

    fig, ax = plt.subplots(3, gridspec_kw={'height_ratios':[1, 1, 1]}, figsize=(13, 10))
    if title:
        fig.suptitle(title)
    
    for a in analyses:
        ax[0].plot(a['Mean'], label=a[label], linewidth=lwdt)
    ax[0].legend()
    ax[0].set_title("Mean position")

    for a in analyses:
        ax[1].plot(a['speed'], label=a[label], linewidth=lwdt)
    ax[1].set_title("Speed")

    for a in analyses:
        ax[2].plot(a['acc'], label=a[label], linewidth=lwdt)
    ax[2].set_title("acceleration")
    
    fig.tight_layout()
    if save and name:
        plt.savefig("figs/" + name + '.png', format='png')
    plt.show()

def plot_rho(analysis,c=False, t_max=-1):
    plt.figure(figsize=[10,12])
    plt.pcolor(analysis.rho, cmap='binary')
    
    t_max = analysis.rho.shape[0] if t_max ==-1 else t_max
    y = np.arange(t_max)
    
    # plt.plot(analysis.analysis['Median'], y, 'b-', linewidth=2, label="Median")
    plt.plot(analysis.analysis['Mean'], y, color='lime', linestyle='-', linewidth=1, label="Mean")
    
    plt.fill_betweenx(y, analysis.analysis['Mean'] - analysis.analysis['std'], analysis.analysis['Mean'] + analysis.analysis['std'],
                 color='darkgreen', alpha=0.2, label="std")


    plt.xlabel('$x$')
    plt.ylabel('$T$', rotation=0)
    plt.colorbar()
    plt.legend()
    plt.show()

def plot_dist(anas, times, title='', save=False, name='', site_max=-1):
    '''
    Parameters:
        times - Iterable of floats between 0 and 1. The relative time steps to plot.
    '''

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    for ana in anas:
        ana_times = ((ana.rho.shape[0] - 1)*times).astype(np.int32)

        L = ana.rho.shape[1]
        x = np.arange(1,L if site_max == -1 else site_max, dtype=np.int32)
        for t, at in zip(times, ana_times):
            y = ana.rho[at, 1:x[-1]+1]
            ax.plot(x, y, label='p={}, t={}'.format(ana.p, t))
            ax.annotate(str(t), (x[np.argmax(y)],y[np.argmax(y)]))
    if title:
        ax.set_title(title)
    else:
        ax.set_title("L={}, Initial position = {}".format(L, ana.d))
    ax.set_xlabel('Site')
    ax.set_ylabel('Probability', rotation=0)
    ax.legend()
    if save:
        fname = name if name else 'position_distribution_over_t_L{}.png'.format(L)
        plt.savefig('figs/' + fname)
    fig.tight_layout()
    plt.show()
