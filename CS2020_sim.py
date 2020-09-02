# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# The following code is used to obtain the size and power under six different cases:

# %%
import numpy as np
from numpy.linalg import inv
import pickle
import multiprocessing as mp

path = '/home/ubuntu/python/2020/Project_LRL'
# %%
with open(path+'/GSW.pickle', 'rb') as f:
    cvalues = pickle.load(f)['GSW']
cvl0 = np.zeros((12,3))
cvl1 = np.zeros((12,3))
for jj in range(12):
    cvl0[jj,:] = np.quantile(cvalues[:,0,jj],(0.90,0.95,0.99))
    cvl1[jj,:] = np.quantile(cvalues[:,1,jj],(0.90,0.95,0.99))


# %%
def lag_v(x,lag,max_lag):
    n = x.shape[0]
    y = x[max_lag-lag:n-lag,:]
    return y


# %%
def sim_process(case_no,T0,TT,tau0,d,rw,seeds,n):
    # Data Generating Process
    np.random.seed(seeds)
    Z0 = np.zeros((n,T0+TT+tau0))
    eps = np.random.multivariate_normal(np.zeros(n),np.eye(n),T0+TT+tau0)
    pi = np.zeros((n,n))
    if case_no == 1:
        pi[0,0] = 0
        pi[1,1] = 0
        pi[0,1] = 0
    elif case_no == 2:
        pi[0,0] = 0
        pi[1,1] = -1*d
        pi[0,1] = 0
    elif case_no == 3:
        pi[0,0] = -1*d
        pi[1,1] = -1*d
        pi[0,1] = 0
    elif case_no == 4:
        pi[0,0] = 0
        pi[1,1] = -1*d
        pi[0,1] = -1*d
    elif case_no == 5:
        pi[0,0] = -1*d
        pi[1,1] = 0
        pi[0,1] = -1*d
    else:
        pi[0,0] = -1*d
        pi[1,1] = -1*d
        pi[0,1] = -1*d
    for tt in range(2,T0+TT+tau0):
        Z0[:,tt] =  Z0[:,tt-1]+pi@Z0[:,tt-1]+eps[tt,:].T
    Z  = Z0[:,50:].T
    T1 = tau0 
    T2 = T0+tau0
    y = Z[T1-tau0:T2-tau0,:1]
    x = Z[T1:T2,1:2]
    # The Estimation Procedure
    dy0 = y[1:,:1] - y[:-1,:1]
    Ly0 = lag_v(y,1,1)
    Lx0 = lag_v(x,1,1)
    T = len(Ly0)
    ss = int(T- np.ceil(T*rw)+1) # the toal shift of moving the x,  the reason +1 is because tau could be zero
    Ty0 = ss
    Ty1 = T
    W = np.zeros((ss,1))
    for tau in range(ss): #
        Tx0 = Ty0-tau
        Tx1 = Ty1-tau
        Y = dy0[Ty0:Ty1,:1]
        X = np.concatenate((Ly0[Ty0:Ty1,:1],Lx0[Tx0:Tx1,:1]),1)
        bhat = inv(X.T@X)@X.T@Y
        sigu2 = np.mean((Y-X@bhat)**2)
        W[tau,0] = bhat.T@(X.T@X)@bhat/sigu2
    return [np.max(W),int(np.where(W[:,0]==np.max(W[:,0]))[0])]


# %%
TT0 =  [200,500,1000,2000]
N_T = len(TT0)
TT = 50
n = 2
R = 1000
d0 = [0.2,0.5,1]
N_d = len(d0)
rw0 =  [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]
tau0 = 30
N_rw = len(rw0)
size = np.zeros((N_rw,3,N_T,N_d,2))
power0 = np.zeros((N_rw,3,N_T,N_d,4))
power1 = np.zeros((N_rw,3,N_T,N_d,4))
datemean =  np.zeros((N_rw,N_T,N_d,4))
case_no = [1,2,3,4,5,6]
for ii in case_no:
    for n_d in range(N_d):
        d = d0[n_d]
        for n_t in range(N_T):
            T0 = TT0[n_t]
            for rw1 in range(N_rw):
                rw = rw0[rw1]
                GSW = np.zeros((R,1))
                date = np.zeros((R,1))
                p = mp.Pool(processes = 16) 
                seeds = np.random.randint(low=1,high=100000,size=R)
                multi_res = [p.apply_async(sim_process,(ii,T0,TT,tau0,d,rw,seeds[rr],2)) for rr in range(R)]
                p.close()
                p.join()
                GSW[:,0]=np.array(([res.get()[0] for res in multi_res]))
                date[:,0]=np.array(([res.get()[1] for res in multi_res]))
                if ii == 1:
                    size[rw1,:,n_t,n_d,ii-1] = np.sum(GSW>cvl1[rw1,:],0)/R
                elif ii ==2:
                    size[rw1,:,n_t,n_d,ii-1] = np.sum(GSW>cvl0[rw1,:],0)/R
                elif ii ==3:
                    power0[rw1,:,n_t,n_d,ii-3] = np.sum(GSW>cvl0[rw1,:],0)/R
                    power1[rw1,:,n_t,n_d,ii-3] = np.sum(GSW>cvl1[rw1,:],0)/R
                    datemean[rw1,n_t,n_d,ii-3] =np.mean(date)
                elif ii ==4:
                    power0[rw1,:,n_t,n_d,ii-3] = np.sum(GSW>cvl0[rw1,:],0)/R
                    power1[rw1,:,n_t,n_d,ii-3] = np.sum(GSW>cvl1[rw1,:],0)/R
                    datemean[rw1,n_t,n_d,ii-3] =np.mean(date)
                elif ii ==5:
                    power0[rw1,:,n_t,n_d,ii-3] = np.sum(GSW>cvl0[rw1,:],0)/R
                    power1[rw1,:,n_t,n_d,ii-3] = np.sum(GSW>cvl1[rw1,:],0)/R
                    datemean[rw1,n_t,n_d,ii-3] =np.mean(date)
                else:
                    power0[rw1,:,n_t,n_d,ii-3] = np.sum(GSW>cvl0[rw1,:],0)/R
                    power1[rw1,:,n_t,n_d,ii-3] = np.sum(GSW>cvl1[rw1,:],0)/R
                    datemean[rw1,n_t,n_d,ii-3] =np.mean(date)
                
res_dict = {'power0': power0,'power1':power1,'size':size, 'datemean':datemean}

file = open(path+'/sim_res.pickle', 'wb')
pickle.dump(res_dict, file)
file.close()

from sendmail import sendemail
sendemail()           

