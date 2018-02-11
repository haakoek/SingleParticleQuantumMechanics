import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from memory_profiler import profile
from scipy.integrate import simps, trapz

v_ho = lambda x, omega=1: 0.5*omega**2*x**2
v_dw = lambda x, omega=1, l_x=1: -omega**2*l_x*np.abs(x) + l_x**2
v = lambda x, omega=1, l_x=1: v_ho(x, omega=omega) #+ v_dw(x, omega=omega, l_x=l_x)

#Exact

"""
This program solves a two-dimensional eigenproblem (the Schrodinger equation)

	(-1/2 nabla^2 + V(x,y)) phi(x,y) = e phi(x,y) (*)
using a uniform rectangular grid, [-Lx, Lx] x [-Ly, Ly], Lx = Ly, where we use N grid points in each dimension resulting in a total of 
M = N^2 grid points. 

	@nabla^2 = d^2/dx^2 + d^2/dy^2
	@V(x,y) is a confining potential, such as the Harmonic oscillator potential
	@phi(x,y) is the unknown eigenfunctions 
	@e is the unknown eigenvalues

The differentiation operator is discretized by a finite difference approximation and (*) can be written on matrix form 

	H*Phi = Phi*E 

	@H is know the matrix representation of the operator h = (-1/2 nabla^2 + V(x,y)) and has only 5 non-zero diagonals 
	 (see chapter9: http://www.uio.no/studier/emner/matnat/math/MAT-INF4130/h17/book2017.pdf for the exact form of H)
	@Phi is a matrix containing the eigenfunctions in its columns 
	@E = diag(e1,...,eN)

The sparse structure of H enables us to take advantage of sparse library in scipy and we can use Lanczos algorithm (scipy.sparse.linalg.eigs(h, k, which=order)) to diagonalize 
H. 
	@h is the matrix to be diagonalized
	@k is the number of desired eigenvalues
	@which='SM' gives the eigenvalues in ascending order (smallest magnitude)
	@which='LM' --- "" ---- descending order (largest magnitude)


"""

N  = 200
Lx = 5
x  = np.linspace(-Lx,Lx,N+2)
y  = np.linspace(-Lx,Lx,N+2)
V  = np.zeros((N,N))
delta_x = x[1]-x[0]
w  = 1
R  = 2

for i in range(0,N):
	for j in range(0,N):
		V[i,j] = 0.5*w**2*(x[i+1]**2 + y[j+1]**2) +  0.5*w**2*(0.25*R**2 - R*abs(x[i+1]))

n = N**2

a = 0.5

h_diag    =  a*4*np.ones(n)/(delta_x**2) + V.flatten("F")
h_off     = -a*np.ones(n-1)/(delta_x**2)
h_off_off = -a*np.ones(n-N)/(delta_x**2) 

k = 1
for i in range(1,n-1):
	if(i%N == 0):
		h_off[i-1] = 0


h = scipy.sparse.diags([h_diag, h_off, h_off,h_off_off,h_off_off], offsets=[0, -1, 1,-N,N])
t0 = time.time()
epsilon, phi = scipy.sparse.linalg.eigs(h, k=3, which="SM")
t1 = time.time()
print np.sort(epsilon).real



phi0 = phi[:,0].real
phi1 = phi[:,1].real
phi2 = phi[:,2].real

phi0 = np.reshape(phi0, (N, N), order='F')
phi1 = np.reshape(phi1, (N, N), order='F')
phi2 = np.reshape(phi2, (N, N), order='F')


X,Y = np.meshgrid(x[1:N+1],y[1:N+1])

plt.figure(1)
plt.pcolor(X,Y,abs(phi0)**2)
plt.colorbar()

plt.figure(2)
plt.pcolor(X,Y,abs(phi1)**2)
plt.colorbar()

plt.figure(3)
plt.pcolor(X,Y,abs(phi2)**2)
plt.colorbar()


plt.show()
