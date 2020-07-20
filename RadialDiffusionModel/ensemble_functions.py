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
    
def Crank_Nicolson(dt,nt,dL,L,f,Dlist,Q,lbc=None,rbc=None,ltype='d',rtype='d',f_return=1):
    '''
    Code for running the Modified Crank Nicolson numerical scheme for radial diffusion as in (Welling et al, 2012).
    Currently functionality requires Diffusion array to extend for one timestep beyond the final time
    (TO DO: Input temporal boundary condition at final timestep)
    
    A source term Q can be provided with the same shape as D. 
    
    TO DO: Input functionality for loss term.
    
    If no BC's are given we assume Dirichlet with constant left and right boundary.
    
    Boundary conditions allowed on left/right are Dirichlet ('d') and Neumann ('n'). Custom BC values for each timestep
    are allowed. Neumann BCs are calculated at order (dL)^2 (might put in functionality for lower order)
    
    Inputs:
        - dt: timestep (seconds)
        - nt: number of time steps
        - dL: space step
        - L: Grid in space
        - f: Initial conditions
        - D: Diffusion coefficient for each time step (list of nt diffusion coefficients)
        - Q: Source term to be applied in each timestep (list of nt source terms)
        - lbc: Left boundary condition (list of nt values for each time step)
        - rbc: Right boundary condition (list of nt values for each time step)
        - ltype: Type of left boundary condition. Default Dirichlet ('d') can also handle Neumann ('n')
        - rtype: Type of right boundary condition. Default Dirichlet ('d') can also handle Neumann ('n')
        - f_return: Which PSDs to include in seconds in result variable res
        
       Outputs:
        - Final PSD T
        - Array of PSD results at timesteps f_return
        
    
    '''
    T = f.copy()
    
    if lbc==None:
        lbc = [T[0]]*nt
    if rbc == None:
        rbc = [T[-1]]*nt
        
    s = (0.5*dt/dL**2)    
    
    res = []
    res.append(f)
    for n in range(nt):
        D = Dlist[n]
        Dplus = Dlist[n+1]
        
        
        Dl = np.array([L[i]**2 * 0.5*((D[i] + D[i-1])/2 + (Dplus[i] + Dplus[i-1])/2)/(L[i]-0.5*dL)**2 \
               for i in range(1,len(L))])
        Dr = np.array([L[i]**2 * 0.5*((D[i] + D[i+1])/2 + (Dplus[i] + Dplus[i+1])/2)/(L[i]+0.5*dL)**2 \
               for i in range(len(L)-1)])

        Dc = np.array([x+y for x,y in zip(Dl[:-1],Dr[1:])])
        
        Qnew = (Q[n]+Q[n+1])/2
        
        if ltype=='d' and rtype=='d':
        
            A = diags([-s*Dl[1:], 1+s*Dc, -s*Dr[1:]], [-1, 0, 1], shape=(len(L)-2, len(L)-2)).toarray() 
            B1 = diags([s*Dl, 1-s*Dc, s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray()
        
           
            #Input Dirichlet boundary for first row
            b = np.zeros(len(L)-2)
            b[0] = s*Dl[0] * lbc[n]
            b[-1] = s*Dr[-1] * rbc[n]
            B = np.add(np.dot(B1,T),b+ dt*(Qnew[1:-1]))
            T[1:-1] = np.linalg.solve(A,B)
            T[0] = lbc[n]
            T[-1] = rbc[n]
            
            
        elif ltype=='d' and rtype=='n':
            A = np.zeros([len(L)-1,len(L)-1])
            B1 = np.zeros([len(L)-1,len(L)])
            
            A[:-1,:] = diags([-s*Dl[1:], 1+s*Dc, -s*Dr[1:]], [-1, 0, 1], shape=(len(L)-2, len(L)-1)).toarray() 
            B1[:-1,:] = diags([s*Dl, 1-s*Dc, s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray()
            
            A[-1,-1] = 1 + (2 * s * Dl[-1])
            A[-1,-2] = -2 * s * Dl[-1]
            B1[-1,-1] = 1 - (2 * s * Dl[-1])
            B1[-1,-2] = 2 * s * Dl[-1]
            
            b = np.zeros(len(L)-1)
            b[0] = s*Dl[0] * lbc[n]
            b[-1] = 4* s * Dl[-1] * rbc[n] * dL
            B = np.add(np.dot(B1,T),b+ dt*(Qnew[1:]))
            T[1:] = np.linalg.solve(A,B)
            T[0] = lbc[n]
           
            
        elif ltype=='n' and rtype=='d':
            A = np.zeros([len(L)-1,len(L)-1])
            B1 = np.zeros([len(L)-1,len(L)])
            
            A[1:,:] = diags([-s*Dl, 1+s*Dc, -s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L)-1)).toarray() 
            B1[1:,:] = diags([s*Dl, 1-s*Dc, s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray()
            
            A[0,0] = 1 + (2 * s * Dr[0])
            A[0,1] = -2 * s * Dr[0]
            B1[0,0] = 1 - (2 * s * Dr[0])
            B1[0,1] = 2 * s * Dr[0]
            
            b = np.zeros(len(L)-1)
            b[-1] = s*Dr[0] * rbc[n]
            b[0] = -4* s * Dr[0] * lbc[n] * dL
            B = np.add(np.dot(B1,T),b+ dt*(Qnew[:-1]))
            T[:-1] = np.linalg.solve(A,B)
            T[-1] = rbc[n]
           
            
        elif ltype=='n' and rtype=='n':
            A = np.zeros([len(L),len(L)])
            B1 = np.zeros([len(L),len(L)])
            
            A[1:-1,:] = diags([-s*Dl, 1+s*Dc, -s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray() 
            B1[1:-1,:] = diags([s*Dl, 1-s*Dc, s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray()
            
            A[0,0] = 1 + (2 * s * Dr[0])
            A[0,1] = -2 * s * Dr[0]
            B1[0,0] = 1 - (2 * s * Dr[0])
            B1[0,1] = 2 * s * Dr[0]
            
            A[-1,-1] = 1 + (2 * s * Dl[-1])
            A[-1,-2] = -2 * s * Dl[-1]
            B1[-1,-1] = 1 - (2 * s * Dl[-1])
            B1[-1,-2] = 2 * s * Dl[-1]
            
            b = np.zeros(len(L))
            b[-1] = 4* s * Dl[-1] * rbc[n] * dL
            b[0] = -4* s * Dr[0] * lbc[n] * dL
            B = np.add(np.dot(B1,T),b+ dt*(Qnew))
            T = np.linalg.solve(A,B)
        
        if n % f_return == 0:
            res.append(T.copy())
            

    return T,res


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
    
