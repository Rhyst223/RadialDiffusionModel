import warnings
warnings.filterwarnings("ignore")
from ensemble_functions import *
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dL', help='Grid step in L*')
    parser.add_argument('dt', help='Time step in seconds')
    parser.add_argument('L_min', help='Lowest L* value in discretisation')
    parser.add_argument('L_max', help='Highest L* value in discretisation')
    parser.add_argument('nt', help='Number of timesteps in seconds')
    parser.add_argument('kp', help='Kp in Ozeke diffusion coefficient')
    parser.add_argument('--f_return', default=1, help='When to return each PSD in result array')
    parser.add_argument('--lbc', default=None, help='Value of left boundary condition')
    parser.add_argument('--rbc', default=None, help='Value of right boundary condition')
    parser.add_argument('--ltype', default='d', help='Type of left boundary condition. Neumann (n) or Dirichlet (d)')
    parser.add_argument('--rtype', default='d', help='Type of left boundary condition. Neumann (n) or Dirichlet (d)')
    
    options = parser.parse_args()
    
    dl, dt = float(options.dL), float(options.dt)
    L = np.arange(float(options.L_min),float(options.L_max),dl)
    f = PSD(L)    
    
    nt = int(options.nt)

    #Ozeke 2014 DLL in SECONDS, for timesteps one more than you need
    D = [Diff_Coeff(L,int(options.kp),full=False)/86400]*int(nt+1)

    #Source term (for simple experiments I haven't used, but we might want to in the project!)
    Q = [np.zeros(len(L))] *int(nt+1)
    
    if options.lbc==None:
        lbc = None
    else:
        lbc = [float(options.lbc)]*nt
    if options.rbc==None:
        rbc = None
    else:
        rbc = [float(options.rbc)]*nt
   
   
    #Run diffusion model
    final_psd, psd_array = Crank_Nicolson(dt,nt,dl,L,f,D,Q,lbc=lbc,rbc=rbc,ltype=options.ltype,rtype=options.rtype,f_return=int(options.f_return))
    
    #Animate the resuls
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=(min(L), max(L)), ylim=(0, max(f)))

    line = ax.plot(L, psd_array[0], color='k', lw=2)[0]

    def animate(i):
        line.set_ydata(psd_array[1:][i, :])
        ax.set_title('t = ' + str((i+1)*int(options.f_return)) + ' seconds')
        
    anim = FuncAnimation(fig, animate, interval=100, frames=len(psd_array)-1, repeat=True)
    fig.show()
