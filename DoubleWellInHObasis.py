import numpy as np
import sympy as sp
from math import factorial
import sys
import scipy.linalg
import pickle
import time

def N(n,m):
    return np.sqrt(factorial(n)/(np.pi*factorial(n+abs(m))))

def R_nm(n,m,r,a=1):
    return (a*r)**abs(m) * L(n,abs(m),a**2*r**2)*sp.exp(-0.5*a**2*r**2)

def L(n,m,x):
    if(n==0):
        return 1.0
    elif(n==1):
        return -x+1+abs(m)
    elif(n==2):
        return 0.5*(x**2 - 2.0*(abs(m)+2)*x + (abs(m)+1)*(abs(m)+2) )
    elif(n==3):
        return (1.0/6.0)*(-x**3 + 3*(abs(m)+3)*x**2 - 3*(abs(m)+2)*(abs(m)+3)*x + (abs(m)+1)*(abs(m)+2)*(abs(m)+3))
    else:
        print "Not implemented"

def theta_integral2(mj,mk):
    M = mk-mj
    if(abs(int(M)) == 1):
        return 0
    else:
        term1 = 1j*(1.0+sp.exp(1j*np.pi*M))
        term2 = sp.exp(1j*np.pi*M)*M - M - 2.0j*sp.exp(0.5*1j*np.pi*M)
        return -term1*term2/(M**2-1.0)

        
def index_map_spinfree(Nspstates,shellMax):
    indexMap = np.zeros((Nspstates/2,2))
    mlMax = shellMax  
    nDecreaseFlag = 0
    count = 0 #count starts at zero for array purposes
    
    for shell in range(0,shellMax+1):
        nMax = shell/2
        n = 0
        nDecreaseFlag = 0
        
        for ml in range(-shell,shell+1,2):
            indexMap[count,0] = n
            indexMap[count,1] = ml
            count += 1
            
            if( nDecreaseFlag == 0 and n < nMax):
                n+=1
            elif(nDecreaseFlag == 0 and n==nMax):
                nDecreaseFlag += 1
                if(shell%2 == 0):
                    n -= 1
            elif( nDecreaseFlag == 1):
                n -= 1
            else:
                print "nDecreaseFlag broken"
    return indexMap

Nparticles = 2
Omega    = 1
shellMax = 7
Nspstates = (shellMax+1)*(shellMax+2)
Nchannels = 3*(4*shellMax+1)
        
indexMapSpinFree = index_map_spinfree(Nspstates,shellMax)

#We want to expand psi_dw in M HO-functions
M = Nspstates/2
I = np.zeros((M,M))

print "SpatialOrbitals: ", M
w = 1.0
a = np.sqrt(w)
r = sp.Symbol('r')


for j in range(0,M):
    nj, mj = indexMapSpinFree[j]
    R_njmj = R_nm(int(nj),int(mj),r)
    N_njmj = N(int(nj),int(mj))
    for k in range(j,M):
        nk, mk = indexMapSpinFree[k]
        R_nkmk = R_nm(int(nk),int(mk),r)
        N_nkmk = N(int(nk),int(mk))
        theta_int  = a**2 * np.complex128(theta_integral2(int(mj),int(mk)))
        if(abs(mk-mj) != 1):
            R_integral = sp.integrate(R_njmj * R_nkmk * r**2, (r,0,sp.oo)) #One r from r dr dt, and one from r |cost|    
        Integral   = N_nkmk*N_njmj * float(R_integral) * theta_int.real
        I[j,k]     = Integral
        I[k,j]     = Integral
            
#Assemble H = [H_{jk}]
H = np.zeros((M,M))
R = 2.0
for j in range(0,M):
    nj, mj = indexMapSpinFree[j]
    eps_j  = 2.0*nj + abs(mj) + 1.0
    H[j,j] = eps_j + (1.0/8.0) * w**2 * R**2
    for k in range(0,M):
        H[j,k] += -0.5*R*w**2 * I[j,k]
        
epsilon, C = scipy.linalg.eigh(H)
print epsilon


HO_pqrs = np.zeros((M, M, M, M))
with open("Coulomb2d_L=72.dat", "r") as f:
    for row in f.read().split("\n"):
        row = row.split()
        if row:
            p, q, r, s, val = row
            HO_pqrs[int(p), int(q), int(r), int(s)] = float(val)

L = 14 #Number of sufficiently good DW-state
with open("EigenEnergiesDW_spOrbs=%d.pkl" % L, "wb") as f:
    pickle.dump(epsilon[0:L],f)

DW_pqrs = np.zeros((L,L,L,L))
DW_pqrs = np.einsum('ap,bq,gr,ds,abgd->pqrs',C[0:M,0:L],C[0:M,0:L],C[0:M,0:L],C[0:M,0:L],HO_pqrs)            

with open("DWpqrs_spOrbs=%d.pkl" % L, "wb") as f:
    pickle.dump(DW_pqrs, f)