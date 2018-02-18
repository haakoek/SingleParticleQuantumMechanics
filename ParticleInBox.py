import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import simps, trapz

N  = 800

L  = 2
x  = np.linspace(0,2*L,N)
dx = x[1]-x[0]

def box_state(x,L):
	if(x <= L):
		return np.sqrt(2/L)*np.sin(np.pi*x/L)
	elif x > L: 
		return 0

psi0 = np.zeros(N)
L = 2

M = 100

Basis = np.zeros((N,M))

psi0 = np.piecewise(x, [x <= L, x > L], [lambda x: np.sqrt(2.0/L)*np.sin(np.pi*x/L), 0])

for j in range(0,M):
	Basis[:,j] = np.piecewise(x, [x <= 2*L, x > 2*L], [lambda x: np.sqrt(1.0/L)*np.sin((j+1)*np.pi*x/(2.0*L)), 0])


Coeffecients = np.zeros(M)

for j in range(0,M):
	val = simps(psi0*Basis[:,j],x)
	if(abs(val) > 1E-7):
		Coeffecients[j] = simps(psi0*Basis[:,j],x)

psi0_fitted = np.zeros(N)

for j in range(0,M):
	psi0_fitted += Coeffecients[j]*Basis[:,j]

#plot(x,psi0,x,psi0_fitted)
#show()

# Put mmatplotlib in interactive mode for animation
ion()

# Setup the figure before starting animation
fig = figure() # Create window
ax = fig.add_subplot(111) # Add axes
line, = ax.plot( x, abs(psi0_fitted)**2, label='$|\Psi(x,t)|^2$' ) # Fetch the line object

# Also draw a green line illustrating the potential
# Add other properties to the plot to make it more elegant
fig.suptitle("Solution of Schrodinger's equation with potential barrier") # Title of plot
ax.grid('on') # Square grid lines in plot
ax.set_xlabel('$x$ [pm]') # X label of axes
ax.set_ylabel('$|\Psi(x, t)|^2$ [1/pm] and $V(x)$ [MeV]') # Y label of axes
ax.set_xlim([0, 2*L])  # Sets x-axis range
ax.set_ylim([0, 1.1])   # Sets y-axis range
ax.legend(loc='best')   # Adds labels of the lines to the window
draw() # Draws first window

T = 10
t = 0
dt = 10**(-2)
w1 = np.pi**2 / (8.0*L**2)
while t < T:
	t += dt
	# Plot this new state
	psi_t = np.zeros(N,dtype=np.complex128)
	for j in range(0,M):
		psi_t += Coeffecients[j]*Basis[:,j]*np.exp(-1j*(j+1)**2*w1*t)
	line.set_ydata( abs(psi_t)**2 ) # Update the y values of the Psi line
	draw() # Update the plot

show()