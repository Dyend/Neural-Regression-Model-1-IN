# My Utility : auxiliars functions
import pandas as pd
import numpy  as np
import math

#gets miniBatch
def get_miniBatch(i,x,bsize):
    z=x[:,i*bsize:(i+1)*bsize]
    return(z)

#STEP 1: Feed-forward of DAE
def forward_dl(x,w,xr):
    L = len(w) - 1
    n_fw = int(L/2)
    a = {}
    a[0] = x
    for i in range(L):
        x = act_sigmoid(np.dot(w[i], x))
        a[i + 1] = x
    for j in range(n_fw):
        xr = act_sigmoid(np.dot(w[j], xr))
    a[i+2] = xr
    return(a)

# STEP 2: Gradiente via BackPropagation
def grad_bp_dl(a,w,y):
    gradW = {}
    deltas = {}

    L = len(w)
    #SoftGrad
    z = np.dot(w[L-1],a[L])
    an = softmax(z)
    error = (y-an)
    divN = (-1/a[L].shape[1])
    errordotX = np.dot(error,a[L].T)
    gradW[L - 1] = divN * errordotX

    #Costo
    tdotlogan = y * np.log(an)
    costo = divN * np.sum(np.sum(tdotlogan,axis=0, keepdims=True))

    #Grads
    L -= 1
    e = a[L] - a[0]
    deltav = e * deriva_sigmoid(a[L])
    gradW[L - 1] = np.dot(deltav, a[L - 1].T)
    deltas[L - 1] = deltav

    for i in reversed(range(0, L-1)):
        mult = np.dot(w[i+1].T, deltas[i+1])
        deltar = mult * deriva_sigmoid(a[i+1])
        gradW[i] = np.dot(deltar, a[i].T)
        deltas[i] = deltar
    return(gradW, costo)      

# Update DL's Weight with Adam
def updW_Adam(w,P,Q,gW,mu):    
    L = len(w)
    E = 10**-8
    B1 = 0.9
    B2 = 0.999
    for i in range(0, L):
        P[i] = B1*P[i] + (1-B1)*gW[i]
        Q[i] = B2*Q[i] + (1-B2)*np.square(gW[i])
        RaizBetas = math.sqrt(1-B2**(i+1))/(1-B1**(i+1))
        QE = Q[i]+E
        P_div_Qsqrt = P[i]/np.sqrt(QE)
        gAdam = RaizBetas*P_div_Qsqrt
        w[i] -= mu * gAdam
    return(w,P,Q)

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))


# Init.weights of the DL 
def iniWPQ():
    W = load_w_dl("./data/w_dl.npz")
    P = []
    Q = []
    j=0
    for i in W:  
        P.append(mat0(i.shape[0], i.shape[1]))
        Q.append(mat0(i.shape[0], i.shape[1]))
        j += 1
    return(W,P,Q)

def mat0(next,prev):
    v  = np.zeros((next,prev))
    return(v)

#Forward Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))

# Métrica
def metricas(x,y):
    cm = np.zeros((x.shape[0], y.shape[0]))
    Fscore = []
    for real, predicted in zip(x.T, y.T):
        cm[np.argmax(real)][np.argmax(predicted)] += 1
    for index, caracteristica in enumerate(cm):
        TP = caracteristica[index]
        FP = cm.sum(axis=0)[index] - TP
        FN = cm.sum(axis=1)[index] - TP
        recall = TP/(TP+FN)
        presition = TP/(TP+FP)
        Fscore.append(2*(presition*recall)/(presition+recall))
    print("Fscore:")
    n=0
    for i in Fscore:
        print(n, " ", i)
        n += 1
    Fscore = pd.DataFrame(Fscore)
    Fscore.to_csv("./data/metrica_dl.csv", index=False, header=False)
    return(Fscore)
    

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the DL
def load_config():      
    par = np.genfromtxt("./data/param_dl.csv",delimiter=',',dtype=None)    
    par_dl=[]    
    par_dl.append(np.float(par[0])) # Learn rate
    par_dl.append(np.int16(par[1])) # miniBatch Size
    par_dl.append(np.int16(par[2])) # MaxIter    
    return(par_dl)

# Load data 
def load_data_csv(fname):
    x     = pd.read_csv(fname, header = None)
    x     = np.array(x)  
    return(x)

def load_data_csv(fname):

    x = pd.read_csv(fname, header=None)
    x = np.array(x)

    return x


def save_w_dl(W, nfile_w, cost, nfile_sft):
    np.savez(nfile_w, idx=W, dtype=object)
    np.savetxt(nfile_sft, cost, delimiter=",", fmt="%.6f")


def load_w_dl(nfile):
    weigth = np.load(nfile, allow_pickle=True)
    w = weigth["idx"]
    return w[()]
