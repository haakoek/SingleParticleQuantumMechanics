import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from scipy.integrate import simps, trapz

Lx    = 8
N     = 200
omega = 1
R     = 2

v_ho = lambda x, omega=1: 0.5*omega**2*x**2
v_dw = lambda x, omega=1, R=1: v_ho(x,omega)+0.5*omega**2*(0.25*R**2 - R*np.abs(x))

HO_groundstate_exact = lambda x, omega=1: (omega/np.pi)**(0.25)*np.exp(-0.5*omega*x**2)

x       = np.linspace(-Lx, Lx, N)
delta_x = x[1] - x[0]
hHO     = np.zeros((N - 2, N - 2)) #Solving for internal grid points only

for i in range(N - 2):
    hHO[i, i] = 1.0/(delta_x**2) + v_ho(x[i + 1], omega=omega)
    if i + 1 < N - 2:
        hHO[i + 1, i] = -1.0/(2*delta_x**2)
        hHO[i, i + 1] = -1.0/(2*delta_x**2)
        
epsilonHO, phiHO = scipy.linalg.eigh(hHO)
print("Three first eigenvalues: ", epsilonHO[0:3])

phiHO = 1.0/np.sqrt(delta_x)*phiHO #Ensure that the wavefunction is normalized wrp. to the integral

print("Analytical <phi0|phi0>: ",simps(HO_groundstate_exact(x)*HO_groundstate_exact(x),x))
print("Numerical  <phi0|phi0>: ",simps(phiHO[:,0]*phiHO[:,0],x[1:-1]))

plt.plot(x[1:-1], np.abs(phiHO[:, 0])**2, '.r', x[1:-1], np.abs(HO_groundstate_exact(x[1:-1]))**2)
plt.title("Exact vs. numerical groundstate of the Harmonic oscillator, w=1")
plt.legend(["Numerical groundstate","Exact groundstate"],loc='upper left')
plt.show()