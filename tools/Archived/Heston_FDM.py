#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:24:23 2023

@author: ted
"""
import numpy as np
from scipy import linalg
from scipy import interpolate 
from tools.Heston_COS_METHOD import heston_cosine_method






def make_matrices(S,V,Smin,Smax,Vmin,Vmax,Ns,Nv,Nt):
    N=Ns*Nv
    Vi = np.repeat(V,Ns).reshape(N,1)
    Si = np.tile(S.T,Nv).T
    Ti = np.linspace(0,T,Nt)
    VminB = np.zeros((N,1))
    VmaxB = np.zeros((N,1))
    SmaxB = np.zeros((N,1))
    SminB = np.zeros((N,1))
    
    ds = (Smax-Smin)/Ns
    dv = (Vmax-Vmin)/Nv
    dt = (Tmax-Tmin)/Nt
    loop_range=np.linspace(Ns+1,N,N-Ns,dtype=int)
    
    for i in loop_range:
        if i % Ns == 0 and i!=N:
            SmaxB[i-1] = 1
    
        if i % Ns == 1 and i!=1:
            SminB[i-1]=1
            
    
    NB = np.zeros((N,1))
    
    mat = SminB+SmaxB
    mat[0:Ns]=1
    mat[N-Ns:N]=1
    
    NB = np.where(mat == 0,1,0)
    NB[Ns-1]=1  
    
    Cs = np.ones((N,1))
    Fs = np.zeros((N,1))
    Bs = np.zeros((N,1))

    
    for b in range(2,Nv):
        for k in range(b*Ns-(Ns-3)-1,b*Ns-1):
            Cs[k]=0
    """
    for b in range(2,Nv):
            Cs[b*Ns+1]=0
            Cs[b*Ns]=0
            Cs[b*Ns-1]=0
            Cs[b*Ns-2]=0
            Bs[b*Ns-2]=1
            Fs[b*Ns+1]=1
    
    Cs[:Ns+2]=0
    Cs[N-Ns-2:N]=0
    """
    Fs[Ns+1]=1
    Fs[N-Ns+1]=0
    
    
    Cv = np.zeros((N,1))
    Fv = np.zeros((N,1))
    Bv = np.zeros((N,1))
    csv = np.zeros((N,1))
    for b in range(3,Nv-1):
        
        for k in range(b*Ns-(Ns-2)-1,b*Ns-1):
            Cv[k]=1
    for k in range(Ns+1,2*Ns-1):
        Fv[k]=1
    for k in range(N-Ns-2,N-2*Ns,-1):
        Bv[k]=1
        
    for b in range(3,Nv-1):
        for k in range(b*Ns-(Ns-2)-1,b*Ns):
            csv[k]=1
            
        
    I_s = np.where(Cs==1)[0]
    derS = np.zeros((N,N))
    derSS = np.zeros((N,N))
    derV1 = np.zeros((N,N))
    derV2 = np.zeros((N,N))
    derVV = np.zeros((N,N))
    derSV = np.zeros((N,N))
    
    for k in I_s:
        derS[k,k-1] = -0.5 * Si[k] / ds
        derS[k,k] = 0
        derS[k,k+1] = 0.5 * Si[k] / ds
        derSS[k,k-1] = Vi[k] * (Si[k]**2) / (ds**2)
        derSS[k,k] = -2 *(Si[k]**2) *Vi[k]/ (ds**2)
        derSS[k,k+1] = (Si[k]**2) *Vi[k]/ (ds**2)
    
    I_v = np.where(Cv==1)[0]
    
    for k in I_v:
        derV1[k,k-Ns] = -0.5 / dv
        derV1[k,k] = 0
        derV1[k,k+Ns] = 0.5 / dv
        
        derV2[k,k-Ns] = -0.5 * Vi[k] / dv
        derV2[k,k] = 0
        derV2[k,k+Ns] = 0.5 *Vi[k]/ dv   
        
        derVV[k,k-Ns] = Vi[k] / (dv**2)
        derVV[k,k] = -2 * Vi[k] / (dv**2)
        derVV[k,k+Ns] = Vi[k] / (dv**2)
        
    I_sv = np.where(csv==1)[0]
        
    for k in I_sv:
        derSV[k,k+Ns+1] = Vi[k] * Si[k] / (4*dv*ds)
        derSV[k,k+Ns-1] = -Vi[k] * Si[k] / (4*dv*ds)
        derSV[k,k-Ns-1] = Vi[k] * Si[k] / (4*dv*ds)
        derSV[k,k-Ns+1] = -Vi[k] * Si[k] / (4*dv*ds)
    return derS,derSS,derV1,derV2,derVV,derSV


#%%

N = Ns*Nv


V = (np.linspace(Vmin,Vmax,Nv)).reshape(1,Nv)
S = (np.linspace(Smin,Smax,Ns)).reshape(Ns,1)






d=Vmax/500
delta_eta = (1/Nv) * np.arcsinh(Vmax/d)
v_j = d*np.sinh(np.linspace(0,Nv,Nv)*delta_eta)

c=K/5
delta_xi = (1/Ns) * ( np.arcsinh( (Smax - K)/c) - np.arcsinh(-K/c))

xi = np.arcsinh(-K/c) + np.linspace(0,Ns,Ns)*delta_xi

S_i = K + c*np.sinh(xi)

derS,derSS,derV1,derV2,derVV,derSV = make_matrices(S_i, v_j, Smin, Smax, Vmin, Vmax, Ns, Nv, Nt)

    
L = (r-q) * derS + kappa * theta * derV1 - kappa*derV2 + 0.5 *derSS \
    + 0.5 * (sigma**2) * derVV + rho*sigma*derSV - r * np.eye(N)

u=np.zeros((N,1))
Si = np.tile(S.T,Nv).T

U = np.exp(-(r-q)*T)*np.maximum(0, Si - K)
thet=0.5
dt = (Tmax-Tmin)/Nt

A = np.eye(N) - thet * dt * L
B = np.eye(N) + (1-thet) * dt*L
invA = linalg.inv(A)



th = [1]

for thet in th:
    U = np.maximum(0, Si - K)

    for t in range(19):
        
        u=U
        if thet == 0:
            U = B @ u
            
        elif thet==1:
            U = invA @ u
        
        else:
            U = invA @ B @ u
    U = U.reshape(Ns,Nv)
    
    V = np.linspace(Vmin,Vmax,Nv)
    S = np.linspace(Smin,Smax,Ns)
    Sii, Vii = np.meshgrid(S,V,indexing='ij',sparse=True)
    interp = interpolate.RegularGridInterpolator((S, V),U)
    z_new = interp([101.52,0.05412])

   # print(z_new)
    
    Sii, Vii = np.meshgrid(S,V)
    tck = interpolate.bisplrep(Sii, Vii,U)
    
    z_new = interpolate.bisplev(101.52,0.05412,tck)
    

    print(z_new)

    
#%%

    
def derivatives_nonuniform(S,V,Smin,Smax,Vmin,Vmax,Ns,Nv,Nt):
    N = Ns*Nv
    Vi = np.repeat(V,Ns).reshape(N,1)
    Si = np.tile(S.T,Nv).T
    Ti = np.linspace(0,T,Nt)
    
    ds = (Smax-Smin)/Ns
    dv = (Vmax-Vmin)/Nv
    dt = (Tmax-Tmin)/Nt
    loop_range=np.linspace(Ns+1,N,N-Ns,dtype=int)
    VminB = np.zeros((N,1))
    VmaxB = np.zeros((N,1))
    SmaxB = np.zeros((N,1))
    SminB = np.zeros((N,1))
    
    for i in loop_range:
        if i % Ns == 0 and i!=N:
            SmaxB[i-1] = 1
    
        if i % Ns == 1 and i!=1:
            SminB[i-1]=1
            
    
    NB = np.zeros((N,1))
    for b in range(2,Nv):
        for k in range(b*Ns - (Ns-2) -1,b*Ns-1):
            NB[k] = 1
    
    NB[Ns-1]=1  
    
    
    mat = SminB+SmaxB
    mat[0:Ns]=1
    mat[N-Ns:N]=1
    
   # NB = np.where(mat == 0,1,0)
   
    
    Cs = np.ones((N,1))
    
    for b in range(2,Nv):
           for k in range(b*Ns - (Ns-3)-1,b*Ns-2):
               Cs[k]=1
    
   
    Cv = np.zeros((N,1))
    
    for b in range(3,Nv-1):
        for k in range(b*Ns-(Ns-2)-1,b*Ns-1):
            Cv[k]=1
            

    csv = np.zeros((N,1))
    for b in range(2,Nv):
        for k in range(b*Ns-(Ns-2)-1,b*Ns-1):
            csv[k]=1
            
            
    I = np.where(Cs==1)[0]
    
    return Cs
    for k in I:
        
        ds = Si[k] - Si[k-1]
        derS[k,k-1] = -0.5 * Si[k] / ds
        derSS[k,k-1] = Vi[k] * (Si[k]**2) / (ds**2)
        
        ds = Si[k+1]-Si[k]
        derS[k,k+1] = 0.5 * Si[k] / ds
        derSS[k,k+1] = Vi[k] * (Si[k]**2) / (ds**2)
        
        ds = (Si[k+1] - Si[k-1])/2
        derS[k,k] = 0
        derSS[k,k] = -2 * Vi[k]*(Si[k]**2) / (ds**2)
    
    I = np.where(Cv==1)[0]
    
    for k in I:
        dv = Vi[k]- Vi[k-Ns]
        derV1[k,k-Ns] = -0.5 / dv
        derV2[k,k-Ns] = -0.5 * Vi[k] / dv
        derVV[k,k-Ns] = Vi[k] / (dv**2)

        dv = Vi[k+Ns] - Vi[k]
        derV1[k,k+Ns] = 0.5 / dv
        derV2[k,k+Ns] = 0.5 *Vi[k]/ dv   
        derVV[k,k+Ns] = Vi[k] / (dv**2)
        
        dv = (Vi[k+Ns] - Vi[k-Ns])/2
        derV1[k,k] = 0
        derV2[k,k] = 0
        derVV[k,k] = -2 * Vi[k] / (dv**2)
        
    I_sv = np.where(csv==1)[0]
    
    for k in I_sv:

        
        d_vs = (Si[k+1] - Si[k]) * (Vi[k]- Vi[k-Ns])
        derSV[k,k-Ns+1] = -Vi[k] * Si[k] / (4*d_vs)

        d_vs = (Si[k+1] - Si[k]) * (Vi[k+Ns]- Vi[k])
        derSV[k,k+Ns+1] = Vi[k] * Si[k] / (4*d_vs)

        d_vs = (Si[k] - Si[k-1]) * (Vi[k+Ns]- Vi[k])
        derSV[k,k+Ns-1] = -Vi[k] * Si[k] / (4*d_vs)
        
        d_vs = (Si[k] - Si[k-1]) * (Vi[k]- Vi[k-Ns])
        derSV[k,k-Ns-1] = Vi[k] * Si[k] / (4*d_vs)
    
    return derS,derSS,derV1,derV2,derVV,derSV
#%%



T=0.15
K=100
r= 0.02
q=0.05
rho=-0.9
v0=0.05
sigma=0.3
kappa=1.5
theta=0.04

Smin=0.
Vmin=0.
Tmin=0.

Smax=2*K
Vmax=0.5
Tmax=T

Ns=79
Nv=39
Nt=3000
N=800
L=17

a = heston_cosine_method(101.52,K,T,N,L,r,q,theta,0.05412,sigma,rho,kappa,'c')[0,0]
print('real', a)

Vi = np.linspace(0,Vmax,Nv)
Si = np.linspace(0,Smax,Ns)
dv = (Vmax-Vmin)/Nv    
ds = (Smax-Smin)/Ns  

I = derivatives_nonuniform(Si, Vi, Smin, Smax, Vmin, Vmax, Ns, Nv, Nt)

derS,derSS,derV1,derV2,derVV,derSV = derivatives_nonuniform(Si, Vi, Smin, Smax, Vmin, Vmax, Ns, Nv, Nt)

I = np.eye(Ns*Nv)

A0 = rho * sigma * derSV
A1 = (r-q) * derS + 0.5 * derSS - 0.5*r*I
A2 = kappa*theta*derV1 - kappa*derV2 + 0.5 * (sigma**2) * derVV - 0.5*r*I

S_i_1 = np.tile(S_i.T,Nv).T
    
U = np.maximum(0,S_i_1-K)
Ti = np.linspace(0,T,Nt)


for t in Ti[2:]:
    u=U
    Y0 = (I + dt*(A0+A1+A2)) @ u
    Y1 = linalg.solve(I - thet * dt*A1, Y0 - thet *dt*A1 @ u) 
    Y2 = linalg.solve(I - thet * dt*A2, Y1 - thet *dt*A2 @ u) 
    U=Y2

U = U.reshape(Ns,Nv)
V = np.linspace(Vmin,Vmax,Nv)
S = np.linspace(Smin,Smax,Ns)
Sii, Vii = np.meshgrid(S_i,v_j)
tck = interpolate.bisplrep(Sii, Vii, U)
    
z_new = interpolate.bisplev(101.52,0.05412,tck)


print('ADI',z_new)


Vi = np.linspace(0,Vmax,Nv)
Si = np.linspace(0,Smax,Ns)
U_fdm = np.copy(U)
data = interpolate.RectBivariateSpline(Si,Vi,U)
z_new = data(101.52,0.05412)
print('FDM: ', z_new[0,0])

#%%
derS,derSS,derV1,derV2,derVV,derSV = derivatives_nonuniform(S_i, v_j, Smin, Smax, Vmin, Vmax, Ns, Nv, Nt)

I = np.eye(N)

A0 = rho * sigma * derSV
A1 = (r-q) * derS + 0.5 * derSS - 0.5*r*I
A2 = kappa*theta*derV1 - kappa*derV2 + 0.5 * (sigma**2) * derVV - 0.5*r*I

S_i_1 = np.tile(S_i.T,Nv).T
    
U = np.maximum(0,S_i_1-K)
Ti = np.linspace(0,T,Nt)


for t in Ti:
    
    u=U
    Y0 = (np.eye(N) + dt*(A0+A1+A2)) @ u
    Y1 = linalg.solve(I - thet * dt*A1, Y0 - thet *dt*A1 @ u) 
    Y2 = linalg.solve(I - thet * dt*A2, Y1 - thet *dt*A2 @ u) 
    U=Y2

U = U.reshape(Ns,Nv)
V = np.linspace(Vmin,Vmax,Nv)
S = np.linspace(Smin,Smax,Ns)
Sii, Vii = np.meshgrid(S_i,v_j)
tck = interpolate.bisplrep(Sii, Vii, U)
    
z_new = interpolate.bisplev(101.52,0.05412,tck)


print('ADI',z_new)



#%%
def heston_explicit(S,V,K,T,Ns,Nv,kappa,theta,rho,sigma,r,q,dt,dv,ds):
    
    U = np.empty((Ns,Nv))
    
    for s in range(Ns):
        for v in range(Nv):
            U[s,v] = np.maximum( S[s] - K , 0)
        
    
    for t in range(Nt - 1):
        
        for v in range(Nv-1):
            U[0,v] = 0
            U[Ns-1,v] = np.maximum( Smax - K ,0)
           # U[Ns-1,v] = Smax 

        for s in range(Ns):
           
            U[s,Nv-1] = np.maximum(S[s] - K, 0)
          #  U[s,Nv-1] = S[s]
        u = np.copy(U)
        
        for s in range(1,Ns-1):
            derV = (u[s,1] - u[s,0]) / dv
            derS = 0.5*(u[s+1,0] - u[s-1,0]) /ds
            U[s,0] = u[s,0] + dt * ( -r*u[s,0] + (r-q) * s * derS + kappa*theta*derV) 
            
        u = np.copy(U)
        for s in range(1,Ns-1):
            for v in range(1,Nv-1):
                A = 1 - dt * ( ((s)**2) *(v)*dv + (sigma**2)*(v) / dv  + r)
                B = 0.5 * s*dt * (s*v*dv - r + q)
                C = 0.5 * s*dt * (s*v*dv + r - q)
                D = (dt/(2*dv)) * ( v*sigma**2 -kappa*(theta-v*dv))
                E = (dt/(2*dv)) * ( v*sigma**2 +kappa*(theta-v*dv))
                F = 0.25 * v*s*dt*sigma*rho
                U[s,v] = A * u[s,v] + B*u[s-1,v] + C*u[s+1,v] + D*u[s,v-1] + E*u[s,v+1] \
                    + F*(u[s+1,v+1] + u[s-1,v-1] - u[s+1,v-1] -u[s-1,v+1] )
            
    return U

#%%
import numpy as np
from scipy import interpolate 

def heston_explicit_nonuniform(S,V,K,T,Ns,Nv,kappa,theta,rho,sigma,r,q,dt,dv,ds):
    
    U = np.empty((Ns,Nv))
    U[:,:] = np.maximum(S-K, 0)[:,None]
        
    for t in range(Nt - 1):
        
        U[0,:]=0
        U[Ns-1,:] = np.maximum(Smax-K,0)
        U[:,Nv-1] = np.maximum(S-K,0)
            
    
        u=U.copy()
    
        derV=(u[1:-1,1]-u[1:-1,0])/(V[1]-V[0])
        derS=(u[2:,0]-u[:-2,0])/(S[2:]-S[:-2])
        U[1:-1,0] = u[1:-1,0] + dt*(-r*u[1:-1,0] + (r-q)*S[1:-1] * derS + kappa*theta*derV)
        
        u = U.copy()
        
        derS = 0.5*(u[2:,1:-1] - u[:-2,1:-1]) / (S[2:]-S[:-2])[:,None]
        derV = 0.5*(u[1:-1,2:]-u[1:-1,:-2]) / (V[2:]-V[:-2])[None,:]
        derSS = (u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) / ((S[2:]-S[1:-1]) * (S[1:-1]-S[:-2]))[:,None]
        derVV = (u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2]) / ((V[2:]-V[1:-1]) * (V[1:-1]-V[:-2]))[None,:]
        derSV = (u[2:,2:] + u[:-2,:-2] - u[:-2, 2:] - u[2:,:-2]) \
               / 4 / (S[2:]-S[:-2])[:,None] / (V[2:]-V[:-2])[None,:]
               
        A = 0.5*V[None,1:-1] * (S[1:-1,None]**2) * derSS
        B = rho*sigma*V[None,1:-1]*S[1:-1,None] * derSV
        C = 0.5*(sigma**2)*V[None,1:-1]*derVV
        D = 0.5*(r-q)*S[1:-1, None]*derS
        E = kappa*(theta-V[None,1:-1])*derV
        L = dt * (A+B+C+D+E- r*u[1:-1,1:-1])
        U[1:-1,1:-1] += L
     
            
   
    return U
Vmax=0.5
Vmin=0
Smin=0
T=0.15

K=100.
Smax = 2*K

Vmax=0.5

r= 0.02
q=0.05
rho=-0.9
v0=0.05
sigma=0.3
kappa=1.5
theta=0.04
L=17
Nv=39

Ns=79

Nt = 3000

dt = T/Nt

Vi = np.linspace(0,Vmax,Nv)
Si = np.linspace(0,Smax,Ns)
dv = (Vmax-Vmin)/Nv    
ds = (Smax-Smin)/Ns  

d=Vmax/500
delta_eta = (1/Nv) * np.arcsinh(Vmax/d)
Vi = d*np.sinh(np.linspace(0,Nv,Nv)*delta_eta)

c=K/5
delta_xi = (1/Ns) * ( np.arcsinh( (Smax - K)/c) - np.arcsinh(-K/c))

xi = np.arcsinh(-K/c) + np.linspace(0,Ns,Ns)*delta_xi

Si = K + c*np.sinh(xi)

call = heston_explicit_nonuniform(Si,Vi,K,T,Ns,Nv,kappa,theta,rho,sigma,r,q,dt,dv,ds)

data = interpolate.RectBivariateSpline(Si,Vi,call)
z_new = data(101.52,0.05412)
print('FDM: ', z_new[0,0])

N=240


real_sol = heston_cosine_method(101.52,K,T,N,L,r,q,theta,0.05412,sigma,rho,kappa,'c')[0,0]

print('real: ', real_sol)
