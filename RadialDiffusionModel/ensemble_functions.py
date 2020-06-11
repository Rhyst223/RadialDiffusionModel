#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:42:10 2019

@author: rlt1917
"""

'''
New file to execute ensemble experiments in PARALLEL
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import math
from scipy.sparse import diags
import argparse

import warnings
warnings.filterwarnings("error")


def Diff_Coeff(L,Kp, full=False):
    """
    Function which calculates the diffusion coefficient at an L-Shell Value
    L and Kp index Kp
    """
    DLL_B = 6.62 * 10**(-13) * L**8 * 10**(-0.0327*(L**2) + 0.625*L \
                 - 0.0108*(Kp**2) + 0.499*Kp)
    
    DLL_E = 2.16 * 10**(-8) * (L**6) * 10**(0.217*L + 0.461*Kp)
    
    D = (DLL_B + DLL_E)
    
    if full==True:
        return DLL_B, DLL_E
    else:
        return D
    
def Crank_Nicolson(dt,nt,dL,L,f,Dlist,Q,lbc=None,rbc=None,ltype='d',rtype='d'):
    '''
    Must give D for each time in a list of lists size (nt)
    To do: edit to handle Dirichlet right and Neumann left BC 
    '''
    T = f.copy()
    
    if lbc==None:
        lbc = T[0]
    if rbc == None:
        rbc = T[-1]
        
    s = (0.5*dt/dL**2)    
    
    res = []
    res.append(f)
    for n in range(nt):
        D = Dlist[n]
        Dplus = Dlist[n+1]
        
        
        Dl = np.array([L[i]**2 * 0.5*((D[i] + D[i-1])/2 + (Dplus[i] + Dplus[i-1])/2)/(L[i]-0.5*dL)**2 \
                       if i!=int(len(L)-1) else 2 * L[i]**2 * 0.5*((D[i] + D[i-1])/2 + (Dplus[i] + Dplus[i-1])/2)/(L[i]-0.5*dL)**2 \
                       for i in range(1,len(L))])
        Dr = np.array([L[i]**2 * 0.5*((D[i] + D[i+1])/2 + (Dplus[i] + Dplus[i+1])/2)/(L[i]+0.5*dL)**2 \
                       if i!=int(len(L)-1) else 0 for i in range(1,len(L))])
        Dc = np.array([x+y for x,y in zip(Dl,Dr)])
        
        A = diags([-s*Dl[1:], 1+s*Dc, -s*Dr], [-1, 0, 1], shape=(len(L)-1, len(L)-1)).toarray() 
        B1 = diags([s*Dl[1:], 1-s*Dc, s*Dr], [-1, 0, 1], shape=(len(L)-1, len(L)-1)).toarray() 
        
           
        #Input Dirichlet boundary for first row
        b = np.zeros(len(L)-1)
        if ltype=='d':
            b[0] = 2*s * 0.5*((D[1] + D[0])/2 + (Dplus[1] + Dplus[0])/2) \
                       / (L[1]-0.5*dL)**2 \
                       *L[1]**2 * lbc
        
        if rtype == 'd':         
            b[-1] = 2*s * 0.5*((D[-2] + D[-3])/2 + (Dplus[-2] + Dplus[-3])/2) \
                       / (L[-2]+0.5*dL)**2 \
                       *L[-2]**2 * rbc            
            
        Qnew = (Q[n]+Q[n+1])/2

        Tn = T.copy()
        if rtype == 'd':
            #Change this to drop the ros and column and implement Dirichlet
            B = np.add(np.dot(B1,Tn[1:-1]),b+ dt*(Qnew[1:-1]))
            T[1:-1] = np.linalg.solve(A,B)
        else:
            B = np.add(np.dot(B1,Tn[1:]),b+ dt*(Qnew[1:]))
            T[1:] = np.linalg.solve(A,B)

        res.append(T.copy())

    return T,dt, res, L, dL


def PSD(L, A=9*10**4, B=0.05, mu=4, sig=0.38, gamma=5):
    """
    Function to calculate the initial phase space density profile over L-Shell
    space L.
    
    Inputs:
        L
        A
        B
        mu
        sigma
        gamma
    
    Outputs:
        f -- Initial Phase Space Density Profile
      
    """
    f = np.zeros(len(L))
    
    for i in range(len(L)):
        
        f[i] = A*np.exp(-(L[i]-mu)**2/(2*sig**2)) + 0.5*A*B*(math.erf(gamma*(L[i]-mu))+1)
    
    return f

#def experiment_temporal(nom):
    
#if __name__ == '__main__':
    
