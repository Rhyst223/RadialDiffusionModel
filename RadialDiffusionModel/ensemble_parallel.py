#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:11:01 2019

@author: rlt1917
"""
import warnings
warnings.filterwarnings("ignore")
from ensemble_functions import *
import numpy as np
import pandas as pd
from multiprocessing import Pool
import itertools
import scipy.stats as ss
import argparse

def spawn_ensembles(inputs):
    dt,nt,dl,L,f,d,q,t,seed, dist,var = inputs
    
    np.random.seed(seed)
    
    
    res = []
    for i in range(len(d)):
        noise = 4
        while noise > 3:
            if dist == 'ln':
                eps = np.random.normal(size=int(48/t),scale=var)
            elif dist == 'll':
                eps = np.random.laplace(size=int(48/t),scale=np.log10(3)/np.log10(2))
            elif dist == 'lu':
                eps = np.random.uniform(size=int(48/t),low=-2*np.log10(3),high=4*np.log10(3))
            elif dist == 'lc':
                eps = ss.cauchy.rvs(size=int(48/t),loc=0,scale=np.log10(3)/np.tan(np.pi/4))
            
            noise = np.max(np.abs(eps))
        eps = list(10**(eps))
            
        eps = np.append(np.repeat(eps[:-1],int(nt/48*t)),eps[-1])
        
        newd = d[i][:int(nt/48*t)] + [dd*ee for dd,ee in zip(d[i][int(nt/48*t):],eps)]
        
        res.append(Crank_Nicolson(dt,nt,dl,L,f,newd,q, \
                                 lbc=None,rbc=0,ltype='d',rtype='n')[0])
       
    return res

def spawn_ensembles_spatial(inputs):
    dt,nt,dl,L,f,d,q,t,seed, dist,space = inputs
    
    np.random.seed(seed)
    
    if space == 0:
        res = []
        for i in range(len(d)):
            noise = 4
            while noise > 3:
                eps = np.random.normal(size=int(48/t),scale=2*np.log10(3)/1.34896)
                
                noise = np.max(np.abs(eps))
            eps = list(10**(eps))
                
            eps = np.append(np.repeat(eps[:-1],int(nt/48*t)),eps[-1])
            
            newd = d[i][:int(nt/48*t)] + [dd*ee for dd,ee in zip(d[i][int(nt/48*t):],eps)]
            
            res.append(Crank_Nicolson(dt,nt,dl,L,f,newd,q, \
                                     lbc=None,rbc=0,ltype='d',rtype='n')[0])
    else:
        res = []
        for i in range(len(d)):
            noise = 4
            while noise > 3:
                eps = np.random.normal(size=(int(48/t),len(np.arange(2.5,6+space,space))), \
                                       scale=2*np.log10(3)/1.34896)
                
                noise = np.max(np.abs(eps))
            eps = 10**(eps)
            stretch = int(space/dl)
            if space == 1:
                s = np.array([int(stretch/2), stretch, stretch, stretch, 1])
            else:
                s = np.array(np.append(np.ones(int(len(eps.T)-1))*stretch,1))
            eps = np.repeat(eps,s.astype(int),axis=1)
            eps = np.repeat(eps,np.append(np.ones(len(eps)-1)*int(nt/48*t),1).astype(int),axis=0)
            
            newd = np.append(np.array(d[i][:int(nt/48*t)]),  \
                             np.multiply(np.array(d[i][int(nt/48*t):]),eps), \
                             axis=0).tolist()
            
            res.append(Crank_Nicolson(dt,nt,dl,L,f,newd,q, \
                                     lbc=None,rbc=0,ltype='d',rtype='n')[0])
           
    return res
        
def experiment_temporal(nom,dt,dl):   
    L = np.arange(2.5,6+dl,dl)
    f = PSD(L)    
    nt = int(172800/dt)
    D = [Diff_Coeff(L,3,full=True)[1]/86400]*int(nt+1)
    Q = [np.zeros(len(L))] *int(nt+1)
    
    deterministic = Crank_Nicolson(dt,nt,dl,L,f,D,Q,lbc=None,rbc=0,ltype='d',rtype='n')[0]
    
    times = [1,3,6,12,24]
    
    D_ens = [D]*nom
    
    nblocks = int(47) #INPUT NCPUS
    blocklen = np.floor_divide(len(D_ens), nblocks)
    
    all_t = []
    for t in times: 
        tt = []
        for block in range(nblocks):
            startind = block*blocklen
            if block != nblocks-1: #not last block
                endind = block*blocklen + blocklen
            else:
                endind = len(D_ens) #going past the end of an array in a slice is fine
            tt.append(D_ens[startind:endind]) #chunk times
        inputs = [(dt,nt,dl,L,f,d,Q,t,k,'ln',2*np.log10(3)/1.34896) for d,k in zip(tt,np.arange(len(tt)))]
        
        _p = Pool(processes=47)
        results = _p.map(spawn_ensembles, inputs,chunksize=1)
        _p.close()
        _p.join()
        combined = list(itertools.chain.from_iterable(results))
        
        del inputs
        all_t.append(combined)
        
    all_t.append([deterministic])
        
    return all_t

def experiment_distributions(nom,dt,dl):   
    L = np.arange(2.5,6+dl,dl)
    f = PSD(L)    
    nt = int(172800/dt)
    D = [Diff_Coeff(L,3,full=True)[1]/86400]*int(nt+1)
    Q = [np.zeros(len(L))] *int(nt+1)
    
    deterministic = Crank_Nicolson(dt,nt,dl,L,f,D,Q,lbc=None,rbc=0,ltype='d',rtype='n')[0]
    
    t = 3
    
    dists = ['ln','ll','lu','lc']
    
    D_ens = [D]*nom
    
    nblocks = int(47) #INPUT NCPUS
    blocklen = np.floor_divide(len(D_ens), nblocks)
    
    all_d = []
    for di in dists: 
        tt = []
        for block in range(nblocks):
            startind = block*blocklen
            if block != nblocks-1: #not last block
                endind = block*blocklen + blocklen
            else:
                endind = len(D_ens) #going past the end of an array in a slice is fine
            tt.append(D_ens[startind:endind]) #chunk times
        inputs = [(dt,nt,dl,L,f,d,Q,t,k,di,2*np.log10(3)/1.34896) for d,k in zip(tt,np.arange(len(tt)))]
        
        _p = Pool(processes=47)
        results = _p.map(spawn_ensembles, inputs,chunksize=1)
        _p.close()
        _p.join()
        combined = list(itertools.chain.from_iterable(results))
        
        del inputs
        all_d.append(combined)
        
    all_d.append([deterministic])
        
    return all_d

def experiment_variances(nom,dt,dl):   
    L = np.arange(2.5,6+dl,dl)
    f = PSD(L)    
    nt = int(172800/dt)
    D = [Diff_Coeff(L,3,full=True)[1]/86400]*int(nt+1)
    Q = [np.zeros(len(L))] *int(nt+1)
    
    deterministic = Crank_Nicolson(dt,nt,dl,L,f,D,Q,lbc=None,rbc=0,ltype='d',rtype='n')[0]
    
    t = 3
    
    var = [2*np.log10(2)/1.34896,2*np.log10(3)/1.34896, 2*np.log10(6)/1.34896,2*np.log10(10)/1.34896]
    
    D_ens = [D]*nom
    
    nblocks = int(47) #INPUT NCPUS
    blocklen = np.floor_divide(len(D_ens), nblocks)
    
    all_v = []
    for v in var: 
        tt = []
        for block in range(nblocks):
            startind = block*blocklen
            if block != nblocks-1: #not last block
                endind = block*blocklen + blocklen
            else:
                endind = len(D_ens) #going past the end of an array in a slice is fine
            tt.append(D_ens[startind:endind]) #chunk times
        inputs = [(dt,nt,dl,L,f,d,Q,t,k,'ln',v) for d,k in zip(tt,np.arange(len(tt)))]
        
        _p = Pool(processes=47)
        results = _p.map(spawn_ensembles, inputs,chunksize=1)
        _p.close()
        _p.join()
        combined = list(itertools.chain.from_iterable(results))
        
        del inputs
        all_v.append(combined)
        
    all_v.append([deterministic])
        
    return all_v

def experiment_spatial(nom,dt,dl):   
    L = np.arange(2.5,6+dl,dl)
    f = PSD(L)    
    nt = int(172800/dt)
    D = [Diff_Coeff(L,3,full=True)[1]/86400]*int(nt+1)
    Q = [np.zeros(len(L))] *int(nt+1)
    
    deterministic = Crank_Nicolson(dt,nt,dl,L,f,D,Q,lbc=None,rbc=0,ltype='d',rtype='n')[0]
    
    t = 3
    
    spaces = [0,1,0.5,0.1]
    
    D_ens = [D]*nom
    
    nblocks = int(47) #INPUT NCPUS
    blocklen = np.floor_divide(len(D_ens), nblocks)
    
    all_s = []
    for space in spaces: 
        tt = []
        for block in range(nblocks):
            startind = block*blocklen
            if block != nblocks-1: #not last block
                endind = block*blocklen + blocklen
            else:
                endind = len(D_ens) #going past the end of an array in a slice is fine
            tt.append(D_ens[startind:endind]) #chunk times
        inputs = [(dt,nt,dl,L,f,d,Q,t,k,'ln',space) for d,k in zip(tt,np.arange(len(tt)))]
        
        _p = Pool(processes=47)
        results = _p.map(spawn_ensembles_spatial, inputs,chunksize=1)
        _p.close()
        _p.join()
        combined = list(itertools.chain.from_iterable(results))
        
        del inputs
        all_s.append(combined)
        
    all_s.append([deterministic])
        
    return all_s

def ensemble_members(nom,dt,dl):   
    L = np.arange(2.5,6+dl,dl)
    f = PSD(L)    
    nt = int(172800/dt)
    D = [Diff_Coeff(L,3,full=True)[1]/86400]*int(nt+1)
    Q = [np.zeros(len(L))] *int(nt+1)
    
    deterministic = Crank_Nicolson(dt,nt,dl,L,f,D,Q,lbc=None,rbc=0,ltype='d',rtype='n')[0]
    
    t = 3
    
    dists = ['ln']
    
    D_ens = [D]*nom
    
    nblocks = int(47) #INPUT NCPUS
    blocklen = np.floor_divide(len(D_ens), nblocks)
    
    all_d = []
    for di in dists: 
        tt = []
        for block in range(nblocks):
            startind = block*blocklen
            if block != nblocks-1: #not last block
                endind = block*blocklen + blocklen
            else:
                endind = len(D_ens) #going past the end of an array in a slice is fine
            tt.append(D_ens[startind:endind]) #chunk times
        inputs = [(dt,nt,dl,L,f,d,Q,t,k,di,2*np.log10(3)/1.34896) for d,k in zip(tt,np.arange(len(tt)))]
        
        _p = Pool(processes=47)
        results = _p.map(spawn_ensembles, inputs,chunksize=1)
        _p.close()
        _p.join()
        combined = list(itertools.chain.from_iterable(results))
        
        del inputs
        all_d.append(combined)
        
    all_d.append([deterministic])
        
    return all_d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='Experiment to run')
    parser.add_argument('members', help='Number if ensemble members')
    parser.add_argument('dt', help='Delta t')
    parser.add_argument('dL', help='Delta L')
    
    options = parser.parse_args()
    
    if options.experiment == 'temporal':
        experiment = experiment_temporal(int(options.members),float(options.dt),float(options.dL))
        pd.DataFrame(experiment).to_pickle('/rds/general/user/rlt1917/home/experiment_temporal.pkl.gz')
    elif options.experiment == 'distributions':
        experiment = experiment_distributions(int(options.members),float(options.dt),float(options.dL))
        pd.DataFrame(experiment).to_pickle('/rds/general/user/rlt1917/home/experiment_distributions.pkl.gz')
    elif options.experiment == 'variances':
        experiment = experiment_variances(int(options.members),float(options.dt),float(options.dL))
        pd.DataFrame(experiment).to_pickle('/rds/general/user/rlt1917/home/experiment_variances.pkl.gz')
    elif options.experiment == 'spatial':
        experiment = experiment_spatial(int(options.members),float(options.dt),float(options.dL))
        pd.DataFrame(experiment).to_pickle('/rds/general/user/rlt1917/home/experiment_spatial.pkl.gz')
    
    elif options.experiment == 'convergence':
        for n in [50,100,150,200,250,300]:
            experiment = ensemble_members(int(n),float(options.dt),float(options.dL))
            pd.DataFrame(experiment).to_pickle('/rds/general/user/rlt1917/home/ensemble_members_'+str(n)+'.pkl.gz')
        
        
    
    
    
    
    
    