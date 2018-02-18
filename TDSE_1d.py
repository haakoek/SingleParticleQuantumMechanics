import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.integrate import simps, trapz
from scipy.misc import derivative
from scipy.linalg import expm

def HarmonicOscillator(x,hw,x0=0):
	return 0.5* (x-x0)**2 * hw**2
def doubleWell(x,d=4,x0=0):
	return (1.0/(2.0*d**2))*(x-x0-0.5*d)**2 * (x-x0+0.5*d)**2
def AnHarmonicOscillator(x,hw,l):
	return 0.5* x**2 * hw**2 + l*x**4
def Laser(x,t,e=0.9,Eps=0.9,tau=2):
    return -x*e*Eps*np.exp(-t**2/tau**2)
    #return -A*np.cos(Omega*t)*x

"""
def Hamiltonian(t,psi,x,w0,w=1.1,Eps0=0.1):
    dx = x[1]-x[0]
    kinetic = 0.5*(-psi[2:N+1] + 2*psi[1:N] - psi[0:N-1])/dx**2
    potential = HarmonicOscillator(x[1:N],hw)*psi[1:N]
    laser = Eps0*Laser(x[1:N],t,w)*psi[1:N]
    return kinetic + potential + laser
"""
def V_func(x,L,V0=15):
	if(4 <= x <= L):
		return V0
	elif(20 <= x <= 20.5):
		return 1000
	else:
		return 0

N  = 200
Lx = 15
L  = 4.75
hw = 1
x  = np.linspace(-Lx,Lx,N+1)
V  = np.zeros(len(x))
dx = x[1]-x[0]
l  = 1

for i in range(0,len(x)):
	V[i] = V_func(x[i],L)

#Compute single-particle basis numerically
H  = np.zeros((N-1,N-1))	
Ht = np.zeros((N-1,N-1))		
for i in range(0,N-1):			
	H[i,i] = 1.0/dx**2 + HarmonicOscillator(x[i+1],hw) #doubleWell(x[i+1])#			
	if(i < N-2):
		H[i,i+1] = -1.0/(2.0*dx**2)
		H[i+1,i] = -1.0/(2.0*dx**2)


eigval, eigvecs = np.linalg.eigh(H)
w0 = eigval[1]-eigval[0]

eigvecs *= 1.0/np.sqrt(dx)

print(eigval[0:3])
print(trapz(eigvecs[:,0]*eigvecs[:,0],x[1:N]))

w = 1

e   = 1
Eps = 1

tau_vec = np.linspace(0,10,100)
plt.plot(tau_vec, e**2 * Eps**2 * np.pi * tau_vec**2 * np.exp(-w**2*tau_vec**2/2) / (2*w)) 
plt.show()
#sys.exit(1)
psi0 = np.zeros(N+1,dtype=np.complex128)
psi_new = np.zeros(N+1,dtype=np.complex128)
potential = np.zeros(len(x))

for i in range(0,len(x)):
	potential[i] = HarmonicOscillator(x[i],hw)

for i in range(0,N-1): 			
	H[i,i] = 1.0/dx**2 + HarmonicOscillator(x[i+1],hw)+V[i+1] #doubleWell(x[i+1])#			

#psi0[1:N] = (1.0/np.sqrt(2.0))*(eigvecs[:,0]+eigvecs[:,1])
psi0[1:N] = eigvecs[:,0]
plt.plot(x,abs(psi0)**2,x,potential) #
plt.axis([-Lx, Lx, 0, 1])
plt.savefig("data/WaveFunc_t=0.0.png")
#plt.axis([-Lx,Lx,0,0.1])

dt = 10**(-2)
Nt = 1000

for i in range(1,int(Nt)+2):
	
    t = i*dt
    #print("t: ",t)

    np.fill_diagonal(Ht,Laser(x[1:N],t))

	
    Htilde = -1j*(H+Ht)
    psi_new[1:N] = np.dot(expm(dt*Htilde),psi0[1:N])
    psi0 = psi_new
	
    if(i%20==0):
		#print (np.vdot(psi0,psi0))
        print (trapz(np.conj(psi0)*psi0,x), t)
        plt.figure(i)
        plt.plot(x,abs(psi0)**2,x,potential) #doubleWell(x)
        plt.axis([-Lx, Lx, 0, 1])
        plt.savefig("data/WaveFunc_t=%g.png" % t)
