import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.integrate import simps, trapz
from scipy.misc import derivative
from scipy.linalg import expm


def HarmonicOscillator(x,hw,x0=0,L=4.75):
	if(x<L):
		return 0.5* (x-x0)**2 * hw**2
	else:
		return 0
def doubleWell(x,d=4,x0=0):
	return (1.0/(2.0*d**2))*(x-x0-0.5*d)**2 * (x-x0+0.5*d)**2
def AnHarmonicOscillator(x,hw,l):
	return 0.5* x**2 * hw**2 + l*x**4
def Laser(x,t,Omega=1,A=-2):
	return A*np.sin(Omega*t)*x
def Hamiltonian(t,psi,x,Omega=1,Eps0=1):
	dx = x[1]-x[0]
	return 0.5*(-psi[2:N+1] + 2*psi[1:N] - psi[0:N-1])/dx**2 + HarmonicOscillator(x[1:N],hw)*psi[1:N] + Eps0*Laser(x[1:N],t,Omega)*psi[1:N]
def V_func(x,L,V0=15):
	if(4 <= x <= L):
		return V0
	elif(20 <= x <= 20.5):
		return 1000
	else:
		return 0

N  = 200
Lx = 20
L  = 4.75
hw = 1
x  = np.linspace(-15,25,N+1)
V  = np.zeros(len(x))
dx = x[1]-x[0]
l  = 1

for i in range(0,len(x)):
	V[i] = V_func(x[i],L)

#Compute single-particle basis numerically
H  = np.zeros((N-1,N-1))	
Ht = np.zeros((N-1,N-1))		
for i in range(0,N-1):			
	H[i,i] = 1.0/dx**2 + HarmonicOscillator(x[i+1],hw,L=1000) #doubleWell(x[i+1])#			
	if(i < N-2):
		H[i,i+1] = -1.0/(2.0*dx**2)
		H[i+1,i] = -1.0/(2.0*dx**2)


eigval, eigvecs = np.linalg.eigh(H)
Omega = eigval[1]-eigval[0]

psi0 = np.zeros(N+1,dtype=np.complex128)
psi_new = np.zeros(N+1,dtype=np.complex128)
potential = np.zeros(len(x))

for i in range(0,len(x)):
	potential[i] = HarmonicOscillator(x[i],hw)

for i in range(0,N-1): 			
	H[i,i] = 1.0/dx**2 + HarmonicOscillator(x[i+1],hw)+V[i+1] #doubleWell(x[i+1])#			

psi0[1:N] = (1.0/np.sqrt(2.0))*(eigvecs[:,0]+eigvecs[:,1])
#psi0[1:N] = eigvecs[:,0]
#plt.plot(x,abs(psi0)**2,x,V,x,HarmonicOscillator(x,hw),x,V) #
#plt.axis([-Lx,Lx,0,0.1])

dt = 10**(-1)
Nt = 150

for i in range(0,int(Nt)+1):
	
	t = i*dt

	if(t<=9):
		np.fill_diagonal(Ht,Laser(x[1:N],t,Omega=Omega))
	else:
		np.fill_diagonal(Ht,Laser(x[1:N],0,Omega=Omega))
	
	Htilde = -1j*(H+Ht)
	psi_new[1:N] = np.dot(expm(dt*Htilde),psi0[1:N])
	psi0 = psi_new
	
	if(i%2==0):
		print np.vdot(psi0,psi0)
		print trapz(np.conj(psi0)*psi0)
		plt.figure(i)
		plt.plot(x,abs(psi0)**2,x,V,x,potential) #doubleWell(x)
		plt.axis([-15, 25, 0, 0.1])
		plt.savefig("data/WaveFunc_t=%.2f.png" % t)
