import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import simps, trapz
import sys

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


w1 = np.pi**2 / (8.0*L**2)

fig, ax = plt.subplots()
line, = ax.plot(x, psi0_fitted)



def propagate(t):
	#t = step*dt
	print t
	psi_t = np.zeros(N,dtype=np.complex128)
	for j in range(0,M):
		psi_t += Coeffecients[j]*Basis[:,j]*np.exp(-1j*(j+1)**2*w1*t)
	line.set_ydata(np.abs(psi_t)**2)  # update the data
	return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

tau = 2*np.pi/w1
time_vec = np.linspace(0,tau/2,100)

ani = animation.FuncAnimation(fig, propagate, time_vec, init_func=init, interval=200, blit=True,repeat=False)
plt.show()