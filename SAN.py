from __future__ import division

import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import animation
import math


def FE(fhand,t,dt,x_0):
    x_t = x_0
    for i in range(int(t/dt)):
        F = fhand(x_t)
        x_next = x_t + dt*F.T[0]
        x_t = x_next
    return x_t

def cellfunction(x):
    
    ######################################
    # constants are here
    
    channel_types = ['Na','K','Le']
    c_m = 1                    # membrane capacitance (Farade)
    g_gj = 0.5            # gap junction conductance
    g = [120, 36, 0.3]    # ion channels conductances (Siemens)
    #z = [1, 1, 2]                    # ion charge
    #e = 1.6e-19
    #Na = 6e23
    #R = 8.31
    #T = 310
    V_0 = [115, -12, 10.613]     # extracellar ion concentrations (Mole)
    
    ######################################
    
    #print 'cellfunction input:',x
    
    Ntypes = len(channel_types)
    N = len(x)
    Ncells = int(N/(Ntypes+1))
    F = np.zeros((N,1))
    
    alpha_n=lambda v: 0.01*(-v+10)/(np.exp((-v+10)*0.1) - 1) if v!=10 else 0.1
    beta_n= lambda v: 0.125*np.exp(-v*0.0125)
    alpha_m=lambda v: 0.1*(-v+25)/(np.exp((-v+25)*0.1) - 1 ) if v!=25 else 1
    beta_m= lambda v: 4*np.exp(-v/float(18))
    alpha_h=lambda v: 0.07*np.exp(-v*0.05)
    beta_h= lambda v: 1/(np.exp((-v+30)*0.1)+1) if v!=30 else 1
    
    alpha = [alpha_n, alpha_m, alpha_h]
    beta = [beta_n, beta_m, beta_h]
    
    for i in range(Ncells):
        
        index = i*(1+Ntypes)
        
        F[index] += -1./c_m*(g[0]*x[index+2]**3*x[index+3]*(x[index]-V_0[0])+g[1]*x[index+1]**4*(x[index]-V_0[1])+g[2]*(x[index]-V_0[2]))
            
        if i > 0:
            F[index] -= 1./c_m*g_gj*(x[index]-x[index-(Ntypes+1)])
                                     
        if i < Ncells-1:
            F[index] -= 1./c_m*g_gj*(x[index]-x[index+(Ntypes+1)])
                                             
        for j in range(Ntypes):
            F[index+j+1] += alpha[j](x[index])*(1-x[index+j+1])-beta[j](x[index])*x[index+j+1]

    return F

alpha_n=lambda v: 0.01*(-v+10)/(np.exp((-v+10)*0.1) - 1) if v!=10 else 0.1
beta_n= lambda v: 0.125*np.exp(-v*0.0125)
n_inf = lambda v: alpha_n(-v)/(alpha_n(-v)+beta_n(-v))

alpha_m=lambda v: 0.1*(-v+25)/(np.exp((-v+25)*0.1) - 1 ) if v!=25 else 1
beta_m= lambda v: 4*np.exp(-v/float(18))
m_inf = lambda v: alpha_m(-v)/(alpha_m(-v)+beta_m(-v))

alpha_h=lambda v: 0.07*np.exp(-v*0.05)
beta_h= lambda v: 1/(np.exp((-v+30)*0.1)+1)
h_inf = lambda v: alpha_h(-v)/(alpha_h(-v)+beta_h(-v))

v_rest = 0

N_cells = 100

x = np.zeros(N_cells*4) # equilibrium zero conditions

print 'x size:',np.size(x)

N_steps = 4000

dt = 0.025

center = N_cells/2*4

print 'center index: ',center

for i in range(N_cells):
    x[i*4]+=v_rest
    x[i*4+1]+=n_inf(v_rest)
    x[i*4+2]+=m_inf(v_rest)
    x[i*4+3]+=h_inf(v_rest)

a = 20
omega = 2*math.pi*0.01 # 10 Hz
#u = lambda t: a*math.sin(omega*t)
#u = lambda t: a*(np.exp(-(t%10)))
u = lambda t: a*1
b = np.zeros(4*N_cells)
b[center] = 1

df = np.zeros(N_steps)

xs = np.zeros((N_steps,N_cells*4))
xs[0] = x

for i in range(1,N_steps,1):
    
    t = i*dt

    #x = traprule(cellfunction,dt,dt,x)
    x[center] += u(t)*dt
    x = FE(cellfunction,dt,dt,x)

    xs[i] = x

def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    i = i*10 #acceleration
    x = range(N_cells)
    y = [xs[i][4*j] for j in range(N_cells)]
    line.set_data(x, y)
    return line,

fig = plt.figure()
ax = plt.axes(xlim=(0, N_cells), ylim=(-50, 100))
line, = ax.plot([], [], lw=2)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=400, interval=1, blit=True)

plt.show()