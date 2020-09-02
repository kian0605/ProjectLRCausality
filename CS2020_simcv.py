#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import multiprocessing as mp
import pickle

path = '/home/ubuntu/python/2020/Project_LRL'
T = 4000
R = 5000 # the number of replications 
rw0 = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]
N_rw = len(rw0)
GSW = np.zeros((R,2,N_rw)) # to store the critical values

def sim_process(T,rw,seeds):
    np.random.seed(seeds)
    wr = np.int(np.ceil(T*rw)) # window size
    Ty0 = T-wr+1
    Ty1 = T
    W1 = -float('inf')
    W0 = -float('inf')
    ux = np.random.randn(T)
    uy = np.random.randn(T)
    ux[0] = 0
    uy[0] = 0
    Wx = np.cumsum(ux)/np.sqrt(T) 
    Wy = np.cumsum(uy)/np.sqrt(T) 
    dWy = Wy[1:]-Wy[0:-1]
    dWx = Wx[1:]-Wx[0:-1]
    for tau in range(0,T-wr):
        Tx0 = Ty0-tau
        Tx1 = Ty1-tau
        E1 = np.zeros((2,1))
        E2 = np.zeros((2,2))
        E1[0,0] = 0.5*(Wy[Ty1-1]**2-Wy[Ty0-1]**2-rw) # is equivalent to np.sum(Wy[Ty0-1:Ty1-2]*dWy[Ty0:])
        E1[1,0] = np.sum(Wx[Tx0-1:Tx1-1]*dWy[Ty0-1:])
        E2[0,0] = np.sum((Wy[0:Ty1]**2)*(1/T))-np.sum((Wy[0:Ty0]**2)*(1/T))
        E2[0,1] = np.sum((Wy[tau:Ty1]*Wx[0:Tx1])*(1/T))-np.sum((Wy[tau:Ty0]*Wx[0:Tx0])*(1/T))
        E2[1,0] = E2[0,1]
        E2[1,1] = np.sum((Wx[0:Tx1]**2)*(1/T))-np.sum((Wx[0:Tx0]**2)*(1/T))
        W1 = np.max([W1,E1.T@np.linalg.inv(E2)@E1])
        #W0 = np.max([W0,])
        W0 = np.max([W0,np.sum(np.sqrt(T)*dWy[-wr:T-1]*dWx[-wr-tau:T-1-tau])**2/rw+E1[0,0]**2/E2[0,0]])
    return [W0,W1]

for rw1 in range(N_rw):
    rw = rw0[rw1]
    p = mp.Pool(processes = 16) 
    seeds = np.random.randint(low=1,high=100000,size=R)
    multi_res = [p.apply_async(sim_process,(T,rw,seeds[rr])) for rr in range(R)]
    p.close()
    p.join()  
    GSW[:,0,rw1]=np.array(([res.get()[0] for res in multi_res]))
    GSW[:,1,rw1]=np.array(([res.get()[1] for res in multi_res]))

    
GSW_dict = {'GSW': GSW}

file = open(path+'/GSW_20200902.pickle', 'wb')
pickle.dump(GSW_dict, file)
file.close()

#from sendmail import sendemail
#sendemail()