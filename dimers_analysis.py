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

@dataclass
class Experiment: 
    file_name : str
    dir_name : str
    results : list
    description : str = ''
    
    def save(self):
        with open(self.dir_name + self.file_name + ".pickle", 'wb') as f:
            pickle.dump(self, f)

@dataclass
class Analysis:   
    L : int
    times: int
    d : int
    batch : int
    p : float
    rho : np.ndarray
    file_name : str
    dir_name : str
    analysis: dict = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.analyze()
        
    def save(self):
        with open(self.dir_name + self.file_name + ".pickle", 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(self.dir_name + self.file_name + ".pickle", 'rb') as f:
            return pickle.load(f)

    def analyze(self):
        print("Analysis start")
        self.analysis['d'] = self.d
        self.analysis['rho'] = self.rho
        self.analysis['batch'] = self.batch
        self.analysis['times'] = self.times
        self.analysis['L'] = self.L
        
        self.analysis['Median'] = 1 + np.sum((np.cumsum(self.rho[:,1:],axis=1)<0.5).astype(int),axis=1).reshape(self.rho.shape[0])
        sites = [np.arange(1, self.rho.shape[1])]

        self.analysis['Mean'] = np.average(np.repeat(sites,self.rho.shape[0],axis=0), axis=1, weights=self.rho[:, 1:]).reshape(self.analysis['Median'].shape)
        self.analysis['std'] = np.sqrt(np.average((np.repeat(sites, self.rho.shape[0], axis=0) -                        self.analysis['Mean'].reshape(self.rho.shape[0], 1))**2 , axis=1, weights=self.rho[:, 1:])).reshape(self.analysis['Median'].shape)
        self.analysis['speed'] = self.analysis['Mean'][1:] - self.analysis['Mean'][:-1]
        self.analysis['acc'] = self.analysis['speed'][1:] - self.analysis['speed'][:-1]
        print("Analysis end")
        return self.analysis

def gaussian(t, a, b):
    return (1/(b*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((t-a)/b)**2)

def exponential(t, a):
    return a*np.exp(-a*t)

def dist_fit(ana, fit, t, p0=None):
    #print(ana.rho[t,1:] != 0)
    #print(np.argwhere(ana.rho[t,1:] != 0))
    x_max, x_min = np.argwhere(ana.rho[t,1:] != 0)[-1][0]+5, np.argwhere(ana.rho[t,1:] != 0)[0][0] - 5
    x_max, x_min = min([x_max, ana.rho.shape[1]]), max([x_min, 1])
    #print(x_max, x_min)
    popt, pcov = curve_fit(fit, np.arange(x_min, x_max), ana.rho[t,x_min:x_max], bounds=(0, x_max),p0=p0)
    #print("L={}, ".format(ana.analysis['L']), "t={}: ".format(t), "Mean = {}, ".format(popt[0]),
    #  "Width = {}".format(popt[1]))
    return popt, pcov, x_max, x_min

def plot_analyses(analyses, label, save=False, title='', name='', log_scale_x=False, log_scale_y=False, t_max=-1):
    lwdt = 1

    fig, ax = plt.subplots(3, gridspec_kw={'height_ratios':[1, 1, 1]}, figsize=(13, 10))
    if title:
        fig.suptitle(title)
    
    for a in analyses:
        a_label = "{}=".format(label) + str(a.analysis[label])
        ax[0].plot(a.analysis['Mean'][:t_max], label=a_label, linewidth=lwdt)
        x = len(a.analysis['Mean'])//2
        y = a.analysis['Mean'][x]
        ax[0].annotate(a_label, (x,y))
    ax[0].legend()
    ax[0].set_title("Mean position")

    for a in analyses:
        ax[1].plot(a.analysis['speed'][:t_max], label="{}=".format(label) + str(a.analysis[label]), linewidth=lwdt)
    ax[1].set_title("Speed")

    for a in analyses:
        ax[2].plot(a.analysis['acc'][:t_max], label="{}=".format(label) + str(a.analysis[label]), linewidth=lwdt)
    ax[2].set_title("acceleration")
    
    fig.tight_layout()
    if log_scale_x:
        ax[0].set_xscale("log", base=log_scale_x)
        ax[1].set_xscale("log", base=log_scale_x)
        ax[2].set_xscale("log", base=log_scale_x)
    if log_scale_y:
        ax[0].set_yscale("log", base=log_scale_y)
        ax[1].set_yscale("log", base=log_scale_y)
        ax[2].set_yscale("log", base=log_scale_y)
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

def plot_dist(ana, times, title='', save=False, name='', site_max=-1):
    '''
    Parameters:
        times - Iterable of floats between 0 and 1. The relative time steps to plot.
    '''

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ana_times = (ana.times*times).astype(np.int32)

    L = ana.rho.shape[1]
    x = range(1,L if site_max == -1 else site_max)
    for t, at in zip(times, ana_times):
        y = ana.rho[at, 1:site_max]
        ax.plot(x[:site_max], y, label='t={}'.format(t))
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
