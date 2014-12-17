from __future__ import division

import matplotlib
matplotlib.use('TKAgg')

from matplotlib import animation
from matplotlib import gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

possiblePresentationModes = ['atrium', 'tissue']
presentationMode = 'atrium'

# Blocking parameter
blockUpperChannel = True

# width
a = 15

# height
b = 15

N_cells = a*b
N_steps = 100

if (N_cells % 2 == 0):
  center = 0 #center = ((N_cells)/2)*4
else:
  center = 0 #center = ((N_cells-1)/2)*4

print 'center index: ',center

def formSAtoAVpath():
    horiz1 = []
    vert1 = []
    passage = []
    if (not blockUpperChannel):
        passage = [8, 9, 10]
    for i in range(b):
        for j in range(a):
            if (((i==0) & (j not in [0, 1])) |
                ((i==1) & (j not in [0, 1])) |
                ((i==2) & (j not in [1, 2])) |
                ((i==3) & (j not in [2, 3])) |
                ((i==4) & (j not in [3, 4, 5, 6])) |
                ((i==5) & (j not in [4, 5, 6, 7, 8, 9])) |
                ((i==6) & (j not in [3, 4, 7, 8, 9])) |
                ((i==7) & (j not in [2, 3]) & (j not in passage)) |
                ((i==8) & (j not in [2, 3, 4, 9, 10, 11])) |
                ((i==9) & (j not in [3, 4, 5, 6, 10, 11, 12])) |
                ((i==10) & (j not in [4, 5, 6, 7, 11, 12, 13])) |
                ((i==11) & (j not in [7, 8, 9, 12, 13, 14])) |
                ((i==12) & (j not in [8, 9, 10, 13, 14])) |
                ((i==13) & (j not in [9, 10, 11, 12, 13, 14])) |
                ((i==14) & (j not in [12, 13, 14]))):
                horiz1.append(i)
                vert1.append(j)
    return horiz1, vert1

def turnOffCells(horiz, vert):
    global cells_turned_off
    cells_turned_off = []
    for i in range(len(horiz)):
        cells_turned_off.append(convertToIndexFromTuple(horiz[i], vert[i]))
    return cells_turned_off

prob = 0.7

def convertToIndexFromTuple(A, B):
  return A*a+B

# Specify the numbers of cell which you want to turn off
# cells_turned_off = [convertToIndexFromTuple(3,4), convertToIndexFromTuple(3,3), convertToIndexFromTuple(3,5)]

def convertToTupleFromIndex(I):
  B = I-int(I/a)*a
  A = int(I/a)
  return (A,B)

# Specify the numbers of cell which you want to turn off
setRandom = False

cells_turned_off = []

if setRandom :
    for i in range(a):
        for j in range(b):
            if (np.random.rand() > prob) & (convertToIndexFromTuple(i, j) != center/4):
                cells_turned_off.append(convertToIndexFromTuple(i, j))

else:
    horiz, vert = formSAtoAVpath()

    #horiz = [0, 2, 1, 3, 3, 0, 1, 2]
    #vert = [2, 0, 2, 0, 1, 3, 3, 3]

    cells_turned_off = turnOffCells(horiz, vert)

alpha_n=lambda v: 0.01*(-v+10)/(np.exp((-v+10)*0.1) - 1) if v!=10 else 0.1
beta_n= lambda v: 0.125*np.exp(-v*0.0125)
n_inf = lambda v: alpha_n(-v)/(alpha_n(-v)+beta_n(-v))

alpha_m=lambda v: 0.1*(-v+25)/(np.exp((-v+25)*0.1) - 1 ) if v!=25 else 1
beta_m= lambda v: 4*np.exp(-v/float(18))
m_inf = lambda v: alpha_m(-v)/(alpha_m(-v)+beta_m(-v))

alpha_h=lambda v: 0.07*np.exp(-v*0.05)
beta_h= lambda v: 1/(np.exp((-v+30)*0.1)+1)
h_inf = lambda v: alpha_h(-v)/(alpha_h(-v)+beta_h(-v))

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
    Ncells = a*b
    F = np.zeros((N,1))
    
    #alpha_n=lambda v: 0.01*(-v+10)/(np.exp((-v+10)*0.1) - 1) if v!=10 else 0.1
    #beta_n= lambda v: 0.125*np.exp(-v*0.0125)
    #alpha_m=lambda v: 0.1*(-v+25)/(np.exp((-v+25)*0.1) - 1 ) if v!=25 else 1
    #beta_m= lambda v: 4*np.exp(-v/float(18))
    #alpha_h=lambda v: 0.07*np.exp(-v*0.05)
    #beta_h= lambda v: 1/(np.exp((-v+30)*0.1)+1) if v!=30 else 1
    
    alpha = [alpha_n, alpha_m, alpha_h]
    beta = [beta_n, beta_m, beta_h]
    
    for i in range(Ncells):
        
        index = i*(1+Ntypes)

        if (i in cells_turned_off):
          F[index] = 0
          continue
        
        # General channel properties for all cells
        F[index] += -1./c_m*(g[0]*x[index+2]**3*x[index+3]*(x[index]-V_0[0])+g[1]*x[index+1]**4*(x[index]-V_0[1])+g[2]*(x[index]-V_0[2]))
        
        # Connecting cells
        if (i%a != 0) and ((index-(Ntypes+1))/4 not in cells_turned_off):   #left neighbour
            F[index] -= 1./c_m*g_gj*(x[index]-x[index-(Ntypes+1)])
        if (i%a != (a-1)) and ((index+(Ntypes+1))/4 not in cells_turned_off):    #right
            F[index] -= 1./c_m*g_gj*(x[index]-x[index+(Ntypes+1)])
        if (i >= a) and ((index-a*(Ntypes+1))/4 not in cells_turned_off):     #up
            F[index] -= 1./c_m*g_gj*(x[index]-x[index-a*(Ntypes+1)])
        if (i < Ncells-a) and ((index+a*(Ntypes+1))/4 not in cells_turned_off):     #down
            F[index] -= 1./c_m*g_gj*(x[index]-x[index+a*(Ntypes+1)])
    
        # ???
        for j in range(Ntypes):
            F[index+j+1] += alpha[j](x[index])*(1-x[index+j+1])-beta[j](x[index])*x[index+j+1]

        if i==16:
            print "dU(16) =", F[index]

    return F

v_rest = 0
dt = 0.05

x = np.zeros(N_cells*4) # equilibrium zero conditions

print 'x size: ',np.size(x)

for i in range(N_cells):
    x[i*4]+=v_rest
    x[i*4+1]+=n_inf(v_rest)
    x[i*4+2]+=m_inf(v_rest)
    x[i*4+3]+=h_inf(v_rest)

amp = 30
omega = 2*math.pi*0.001 # 1 Hz
#u = lambda t: a*math.sin(omega*t)
u = lambda t: 10*a*(np.exp(-(t%10)))
#u = lambda t: amp*1

xs = np.zeros((N_steps, N_cells*4))
xs[0] = x

for i in range(1, N_steps, 1):
    t = i*dt

    #x = traprule(cellfunction,dt,dt,x)
    x[center] += u(t)*dt
    x = FE(cellfunction,dt,dt,x)

    xs[i] = x

def initGraph():
    line.set_data([], [])
    return line,

def initGrid():
    mat.set_data(gridAtTimestep(0))
    return mat,

def initFinal():
    line.set_data([], [])
    mat.set_data(gridAtTimestep(0))
    return line,

# animation function.  This is called sequentially
def animateGraph(i):
    #i=i*10 #acceleration
    x = range(N_cells)
    y = [xs[i][4*j] for j in range(N_cells)]
    line.set_data(x, y)
    return line,

def gridAtTimestep(timeStep):
    grid = np.zeros(N_cells).reshape(b, a)
    for i in range(b):
      for j in range(a):
        grid[i][j] = normalizePotential(xs[timeStep][4*(a*i+j)])
    for I in cells_turned_off:
        (A,B) = convertToTupleFromIndex(I)
        grid[A][B] = -100
    return grid

def animateGrid(timeStep):
    timeStep = timeStep #*10
    mat.set_data(gridAtTimestep(timeStep))
    return mat,

def animateFinal(timeStep):
    mat, = animateGrid(timeStep)
    line, = animateGraph(timeStep)
    return line, mat

def normalizePotential(pot):
  #return (pot-(-v_rest))/(100-(-v_rest))
  return pot

# 1 - draw a plot
# 2 - draw a grid
# 3 - draw a static grid
# 4 - FINAL presentation

fig = plt.figure()
mode = 4

if mode == 1:
  # Draw an impulse propagation in a chain
  fig = plt.figure()
  ax = plt.axes(xlim=(0, N_cells-1), ylim=(-30, 120))
  line, = ax.plot([], [], lw=2)

  anim = animation.FuncAnimation(fig, animateGraph, init_func=initGraph,
                               frames=N_steps, interval=2, blit=True)
elif mode == 2:
  # Draw a grid with impulse values
  grid = gridAtTimestep(0)
  ax = fig.add_subplot(111)
  mat = ax.matshow(grid, interpolation='none', vmin=0, vmax=1)
  plt.colorbar(mat)
  ani = animation.FuncAnimation(fig, animateGrid, init_func = initGrid, frames=N_steps, interval=2, blit=True)

elif mode == 3:
  fig, ax = plt.subplots()
  mat = ax.matshow(gridAtTimestep(10), interpolation='none', vmin=0, vmax=1)
  plt.colorbar(mat)
elif mode == 4:
  gs = gridspec.GridSpec(4, 4)
  gs.update(left = 0.07, right = 1.05, wspace = 0.02)

  grid = gridAtTimestep(0)
  ax_grid = fig.add_subplot(gs[:-1, :-2])
  mat = ax_grid.matshow(grid, cmap = cm.RdBu_r, interpolation='none', vmin=-50, vmax=50)
  plt.colorbar(mat)

  ax_line = fig.add_subplot(gs[-1, :-2])
  ax_line.set_xlim([0, N_cells-1])
  ax_line.set_ylim([-50, 100])
  ax_line.set_yticks([-50, 0, 100])
  ax_line.axhline(y=0, ls='--', color='k')
  line, = ax_line.plot([], [], lw=1)

  ax_line2 = fig.add_subplot(gs[-1, -2:])
  ax_line2.set_xlim([0, N_cells-1])
  ax_line2.set_ylim([-50, 100])
  ax_line2.set_yticks([-50, 0, 100])
  ax_line2.axhline(y=0, ls='--', color='k')
  line, = ax_line2.plot([], [], lw=1)

  ax_heart = fig.add_subplot(gs[:-1, -2:])
  arr_image = mpimg.imread('heart2.png')
  ax_heart.imshow(arr_image)
  ax_heart.axes.get_xaxis().set_visible(False)
  ax_heart.axes.get_yaxis().set_visible(False)

  ani = animation.FuncAnimation(fig, animateFinal, init_func = initFinal, frames=N_steps, interval=2, blit=True)

elif mode == 5:
  plt.plot([xs[int(b/2)][4*i] for i in range(a*b)])

plt.show()