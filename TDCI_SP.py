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
def Pert(x,t,e=0.5,Eps=0.5,tau=6):
    return -x*e*Eps*np.exp(-t**2/tau**2)
    #return -A*np.cos(Omega*t)*x
def Laser(t,Omega=1,Eps=1):
	return Eps*np.sin(Omega*t)
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

eigvecs *= 1.0/np.sqrt(dx)

print(eigval[0:3])
print(trapz(eigvecs[:,0]*eigvecs[:,0],x[1:N]))

psi0    = np.zeros(N+1,dtype=np.complex128)
psi_new = np.zeros(N+1,dtype=np.complex128)

States = 4

C    = np.zeros(States)
C[0] = 1

H0 = np.zeros((States,States))
V  = np.zeros((States,States))

for i in range(0,States):
	H0[i,i] = eigval[i]
	for j in range(0,States):
		V[i,j] = np.trapz(eigvecs[:,i]*x[1:N]*eigvecs[:,j],x[1:N])

print V

Nt = 40000
dt = 10**(-3)

for i in range(1,Nt+1):

	t = i*dt
	#print t-dt
	Htilde = -1j*(H0+Laser(t-dt,Omega=eigval[3]-eigval[0])*V)
	C_new = np.dot(expm(dt*Htilde),C)
	C     = C_new
	#C = C - 1j*dt*np.dot(H0+Laser(0)*V,C)

	if(i%100 == 0):	
		psi_t = np.zeros(len(eigvecs[:,0]),dtype=np.complex128)
		for j in range(0,States):
			psi_t += C[j]*eigvecs[:,j]
		print("Norm: ", abs(trapz(np.conj(psi_t)*psi_t,x[1:N]))**2, "t: ", t)
		plt.figure(i)
		plt.plot(x[1:N],np.abs(psi_t)**2,'-b')
		plt.plot(x[1:N],HarmonicOscillator(x[1:N],hw),'-y')
		plt.plot(x[1:N],np.abs(eigvecs[:,0])**2,'-r')
		plt.plot(x[1:N],np.abs(eigvecs[:,1])**2,'-g')
		plt.axis([-Lx, Lx, 0, 1])
		plt.savefig("data/WaveFunc_t=%.2f.png" % t)