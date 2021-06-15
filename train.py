# Deep-Learning:Training via BP+Pseudo-inverse

import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    w = ut.randW(y.shape[0], x.shape[0])
    v = ut.mat0(y.shape[0], x.shape[0])
    costo = []
    for iter in range(param[0]):
        gW,cost = ut.softmax_grad(x,y,w)
        costo.append(cost)
        w,v     = ut.updW_softmax(w,v,gW,param[1])                       
    return(w,costo)

#gets miniBatch
def get_miniBatch(i, x, bsize):
    bsize = int(bsize)
    start_idx = i * bsize
    xe = x[:, start_idx: start_idx + bsize]
    return (xe)

# Deep-AE's Training 
def train_dae(x,W,V,mu,numBatch,BatchSize):
    for i in range(numBatch):        
        xe   = get_miniBatch(i,x,BatchSize)        
        Act  = ut.forward_dae(xe,W)              
        gW   = ut.grad_bp_dae(Act,W) 
        W,V  = ut.updW_dae(W,V,gW,mu)
    return(W)

#Deep Learning: Training 
def train_dl(x,param):
    W,V = ut.ini_WV(x.shape[0], param[3:])   # completar
    numBatch = np.int16(np.floor(x.shape[1]/param[1]))        
    for Iter in range(param[2]):        
        xe  = x[:,np.random.permutation(x.shape[1])]
        W   = train_dae(xe,W,V,param[0],numBatch,param[1])            
    return(W) 
   
# Beginning ...
def main():
    par_dae,par_sft = ut.load_config()    
    xe              = ut.load_data_csv('./data/train_x.csv')    
    ye              = ut.load_data_csv('./data/train_y.csv')    
    W               = train_dl(xe,par_dae) 
    Xr              = ut.encoder(xe,W)
    Ws, cost        = train_softmax(Xr,ye,par_sft)
    ut.save_w_dl(W,Ws,'./data/w_dl.npz',cost,'./data/costo_softmax.csv')
       
if __name__ == '__main__':   
	 main()

