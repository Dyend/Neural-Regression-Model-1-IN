# Fase 2: Deep-Learning:Training via Adam

import pandas     as pd
import numpy      as np
import my_utility as ut

#gets miniBatch
def get_miniBatch(i, x, bsize):
    bsize = int(bsize)
    start_idx = i * bsize
    xe = x[:, start_idx: start_idx + bsize]
    return (xe)
	
#Training: Deep Learning
def train_dl(x,y,param):
    W,P,Q     = ut.iniWPQ()  
    numBatch = np.int16(np.floor(x.shape[1]/param[1]))    
    cost = []
    for Iter in range(param[2]):
        xe  = x[:,np.random.permutation(x.shape[1])]
        for i in range(numBatch):        
            xe   = get_miniBatch(i,x,param[1])
            Act  = ut.forward_dl(xe,W,x)              
            gW,costo   = ut.grad_bp_dl(Act,W,y) 
            W,P,Q  = ut.updW_Adam(W,P,Q,gW,param[0])
            cost.append(costo)
    return(W, cost) 
   
# Beginning ...
def main():
    par_dl          = ut.load_config()
    xe              = ut.load_data_csv('./data/train_x.csv')    
    ye              = ut.load_data_csv('./data/train_y.csv')    
    W, cost         = train_dl(xe,ye,par_dl)         
    ut.save_w_dl(W,'./data/w_dl.npz',cost,'./data/costo_dl.csv')
       
if __name__ == '__main__':   
	 main()

