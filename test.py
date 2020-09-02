#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 07:47:30 2020

@author: kian
"""

import numpy as np
import multiprocessing as mp
T = 4000
R = 10000 # the number of replications 
pi =  [0.4,0.3,0.2,0.1]
N_rw = len(pi)
GSW = np.zeros((R,N_rw)) # to store the critical values

def sim_process(T,pi,seeds):
    np.random.seed(seeds)
    T0 = int(np.ceil(T*pi))
    T1 = int(np.ceil(T*(1-pi)))
    W = -float('inf')
    uy = np.random.randn(T)
    uy[0] = 0
    Wy = np.cumsum(uy)/np.sqrt(T) 
    for tau in range(T0,T1):
        W = np.max([(Wy[tau]-(tau)/T*Wy[-1])**2/((tau)/T*(1-(tau)/T)),W])
    return W

for rw1 in range(N_rw):
    rw = pi[rw1]
    p = mp.Pool(processes = 6) 
    seeds = np.random.randint(low=1,high=100000,size=R)
    multi_res = [p.apply_async(sim_process,(T,rw,seeds[rr])) for rr in range(R)]
    p.close()
    p.join()  
    GSW[:,rw1]=np.array(([res.get() for res in multi_res]))
    
# array([ 8.91777997, 10.30532969, 13.86478062])
# array([17.4210623 , 18.7716589 , 22.11106194])
from numpy.linalg import inv
T = 4000
rw = 0.2
def sim_process2(T,rw,seeds):
    np.random.seed(seeds)
    y = np.random.randn(T,1) 
    x = np.random.randn(T,1) 
    wr = int(np.ceil(T*rw))
    W0 = -float('inf')
    for ss in range(3201):
        Y = y[-wr:T,0:]
        X = x[-wr-ss:T-ss,0:]
        bhat = inv(X.T@X)@X.T@Y
        sigu2 = np.mean((Y-X@bhat)**2)
        W0 = np.max([W0,bhat.T@(X.T@X)@bhat/sigu2])
    return W0

GSW2 = np.zeros((R,1)) # to store the critical values
p = mp.Pool(processes = 6) 
seeds = np.random.randint(low=1,high=100000,size=R)
multi_res = [p.apply_async(sim_process2,(T,0.2,seeds[rr])) for rr in range(R)]
p.close()
p.join()  
GSW2[:,0]=np.array(([res.get() for res in multi_res]))