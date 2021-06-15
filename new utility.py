# My Utility : auxiliars functions
import pandas as pd
import numpy  as np

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))


#STEP 1: Feed-forward of DAE
def forward_dae(x,w):	
    # completar code
    return(a)    
# STEP 2: Feed-Backward
def grad_bp_dae(a,w):
    # completar code
    
    return(gradW, Cost)    

# Update DAE's weight with RMSprop
def updW_dae(w,v,gW,mu):    
    # completar code
        
    return(w,v)
#    
# Update Softmax's weight with RMSprop
def updW_softmax(w,v,gW,mu):    
    # completar code
    
    return(w,v)

# Initialize weights of the Deep-AE
def ini_WV(...):
    # completar code
    return(W)

# Initialize random weights
def randW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    return(w)

#Forward Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))

# Softmax's gradient
def softmax_grad(x,y,w):
    # completar code

    return(gW,Cost)
  
# Encoder
def encoder(x,w):
    #completar code
    
    return(...)
# Métrica
def metricas(x,y):
    # completar code
    return(Fscore)
    

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the SNN
def load_config():      
    par = np.genfromtxt("param_dae.csv",delimiter=',',dtype=None)    
    par_sae=[]    
    par_sae.append(np.float(par[0])) # Learn rate
    par_sae.append(np.int16(par[1])) # miniBatchSize
    par_sae.append(np.int16(par[2])) # MaxIter
    for i in range(3,len(par)):
        par_sae.append(np.int16(par[i]))
    par    = np.genfromtxt("param_softmax.csv",delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning     
    return(par_sae,par_sft)
# Load data 
def load_data_csv(fname):
    x     = pd.read_csv(fname, header = None)
    x     = np.array(x)  
    return(x)

# save weights of the DL in numpy format
#W,Ws,'w_dl.npz',cost,'costo_softmax.csv'
def save_w_dl(W,Ws,nfile_w,cost,nfile_sft):    
    # cmpletar code
    
    
#load weight of the DL in numpy format
def load_w_dl(nfile):
    #completar code    
        
    return(W)    

# save weights in numpy format
def save_w_npy(w1,w2,mse):  
    # completar code