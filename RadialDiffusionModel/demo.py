import warnings
warnings.filterwarnings("ignore")
from ensemble_functions import *
import numpy as np
import pandas as pd
from multiprocessing import Pool
import itertools
import scipy.stats as ss
import argparse

L = np.arange(2.5,6+dl,dl)
f = PSD(L)    
nt = int(172800/dt)

#Ozeke 2014 DLL in SECONDS, for timesteps one more than you need
D = [Diff_Coeff(L,3,full=True)[1]/86400]*int(nt+1)

#Source term (for simple experiments I haven't used, but we might want to in the project!)
Q = [np.zeros(len(L))] *int(nt+1)

#Run diffusion model
diffusion = Crank_Nicolson(dt,nt,dl,L,f,D,Q,lbc=None,rbc=[0]*nt,ltype='d',rtype='n')
