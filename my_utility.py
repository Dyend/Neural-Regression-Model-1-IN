# My Utility : auxiliars functions
import pandas as pd
import numpy  as np

# Calculate Pseudo-inverse
def pinv_ae(a1,x,hn,C):
  xh = np.dot(x,a1.T)
  ai = np.dot(a1,a1.T)+ np.eye(int(hn))/C
  p_inv = np.linalg.pinv(ai)
  w2 = np.dot(xh,p_inv)
  return w2

#AE's Feed-Backward
def backward_ae(act, x, w1, w2, mu):

  e = act[2] - x
  #Cost = np.mean(e**2)/(2*e.shape[1])
  #dOut = e * derivate_act(act[2])
  #gradW2 = np.dot(dOut, act[1].T)
  dHidden = np.dot(w2.T, e) * derivate_act(act[1])
  gradW1 = np.dot(dHidden, act[0].T)
  w1 = w1 - mu * gradW1
  return w1

    
#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def derivate_act(a):
    return(a*(1-a))


#Forward Softmax
def softmax(z):
    z = np.exp(z-np.max(z))
    an = z / z.sum(axis=0, keepdims=True)
    return(an)


# Softmax's gradient
def softmax_grad(w,x,y,lambd):
    #softmax
    z = np.dot(w,x)
    an = softmax(z)

    #Calculo del Costo
    divN = (-1/x.shape[1])
    tdotlogan = y * np.log(an)
    lambdotW = (lambd/2) * np.linalg.norm(w,2)
    cost = divN * np.sum(np.sum(tdotlogan,axis=0, keepdims=True))
    Cost = cost + lambdotW
    #Calculo del Gradiente
    error = (y-an)
    errordotX = np.dot(error,x.T)
    lambadotW = lambd * w
    gradW = divN * errordotX + lambadotW
    print('Costo: ',Cost)
    return(gradW, Cost)


# Initialize weights
def iniW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    return(w)

import sys
#Measure
def metricas(yv,zv):
    zv = (zv == zv.max(axis=0, keepdims=True)).astype(int)
    print('\nPrediccion en binario\n', zv)
    print('\nValor deseado en binario\n', yv)
    CM = pd.crosstab(yv.tolist(),zv.tolist())
    acc = accuracy(CM)
    print('\nAccuracy: ',acc,'\n')
    return()

def accuracy(CM):
    acc = np.diag(CM).sum() / CM.to_numpy().sum()
    return(acc)
#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the DL: 
def load_config():      
    par = np.genfromtxt("./data/param_sae.csv",delimiter=',')    
    par_sae=[]
    par_sae.append(np.float(par[0])) # % train
    par_sae.append(np.float(par[1])) # Learn rate
    par_sae.append(np.int16(par[2])) # Penal. C
    par_sae.append(np.int16(par[3])) # MaxIter
    for i in range(4,len(par)):
        par_sae.append(np.int16(par[i]))
    par    = np.genfromtxt("./data/param_softmax.csv",delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning rate
    par_sft.append(np.float(par[2]))   #Lambda
    return(par_sae,par_sft)
# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    return(x)

# save weights of the DL in numpy 
def save_w_dl(W,Ws,cost,ruta_peso,ruta_costo):
    #Se guardan los pesos en formato npz
    W[len(W)]=Ws
    savez_dict = dict()
    for i in range(len(W)):
        key = str(i)
        savez_dict[key] = W[i] 
    np.savez_compressed(ruta_peso, **savez_dict)
    load_w_dl(ruta_peso)
    #Se guardan los costos en formato csv
    np.savetxt(ruta_costo, np.array(cost),delimiter=',')
    return()  
    

#load weight of the DL in numpy 
def load_w_dl(ruta_peso):
    data = np.load(ruta_peso)
    return(data)    
