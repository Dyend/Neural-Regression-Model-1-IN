# Deep-Learning:Training via BP+Pseudo-inverse

import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    w = ut.iniW(y.shape[0],x.shape[0])
    Costos = []
    print(x)
    print(x.shape)
    for i in range(1,int(param[0])):
        gradW, costo = ut.softmax_grad(w,x,y,param[2])
        w = w - (param[1] * gradW)
        Costos.append(costo)
    return(w,Costos)

# AE's Training 

def train_ae(x,hnode,param):
    #Inicialisamos W1
    w1 = ut.iniW(param[hnode],x.shape[0])
    for i in range(1, param[3]):
        #Obtenemos el el peso W2 con la funcion pinv_ae
        z = np.dot(w1,x)
        a1 = ut.act_sigmoid(z)
        w2 = ut.pinv_ae(a1,x,param[hnode],param[2])

        #Modificamos el peso w1 con la funcion backward_ae
        z2 = np.dot(w2,a1)
        act = []
        act.append(x)
        act.append(a1)
        act.append(z2)
        w1 = ut.backward_ae(act, x, w1, w2, param[1])
        return(w1)

def train_sae(x,param):
    W={}
    for hn in range(4,len(param)):   #Number of AEs     
        w1       = train_ae(x,hn,param)
        W[hn-4]  = w1
        x        = ut.act_sigmoid(np.dot(w1,x))
    return(W,x) 
   
# Beginning ...
def main():
    par_sae,par_sft = ut.load_config()    
    xe              = ut.load_data_csv('./data/train_x.csv')
    ye              = ut.load_data_csv('./data/train_y.csv')
    W,Xr            = train_sae(xe,par_sae)
    Ws, cost        = train_softmax(Xr,ye,par_sft)
    ut.save_w_dl(W,Ws,cost,'./data/w_dl.npz','./data/cost_softmax.csv')
if __name__ == '__main__':   
	 main()

