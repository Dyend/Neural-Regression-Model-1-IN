# My Utility : auxiliars functions
import pandas as pd
import numpy  as np



def forward_dae(x,w):	
    a = {}
    a[0] = x
    for i in range(len(w)):
        x = act_sigmoid(np.dot(w[i], x))
        a[i + 1] = x
    return(a)    


def grad_bp_dae(a,w):
    gradW = {}
    deltas = {}
    L = len(w)
    e = a[L] - a[0]
    deltav = e * deriva_sigmoid(a[L])
    gradW[L - 1] = np.dot(deltav, a[L - 1].T)
    deltas[L - 1] = deltav
    for i in reversed(range(0, L-1)):
        mult = np.dot(w[i+1].T, deltas[i+1])
        deltar = mult * deriva_sigmoid(a[i+1])
        gradW[i] = np.dot(deltar, a[i].T)
        deltas[i] = deltar
    return(gradW)    


#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   


# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))






def iniW(input, nodesEnc):
    W = []
    aux = input
    for i in range(len(nodesEnc)):
        W.append(randW(nodesEnc[i], aux))
        aux = nodesEnc[i]
    for i in reversed(W):
        W.append(randW(i.shape[1], i.shape[0]))
    return(W)


def randW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    return(w)



def mat0(next,prev):
    v  = np.zeros((next,prev))
    return(v)

# Initialize weights of the Deep-AE
def ini_WV(input, nodesEnc):
    W = []
    V = []
    aux = input
    for i in range(len(nodesEnc)):
        W.append(randW(nodesEnc[i], aux))
        V.append(mat0(nodesEnc[i], aux))
        aux = nodesEnc[i]
    for i in reversed(W):
        W.append(randW(i.shape[1], i.shape[0]))
        V.append(mat0(i.shape[1], i.shape[0]))
    return(W,V)

def updW_sgd(w, gradW, mu):
    L = len(w)
    for i in range(0, L):
        tau = mu/len(w)
        mu_k = mu/(1+np.dot(tau, (i+1)))
        w[i] -= mu_k * gradW[i]
    return w


# Update DAE's weight with RMSprop
def updW_dae(w,v,gW,mu):    
    L = len(w)
    E = 10**-10
    B = 0.9
    for i in range(0, L):
        aux = v[i]+E
        mu_k = mu/np.sqrt(aux)
        w[i] -= mu_k * gW[i]
        v[i] = B*v[i] + (1-B)*np.square(gW[i])
    return(w,v)
#    
# Update Softmax's weight with RMSprop
def updW_softmax(w,v,gW,mu):
    E = 10**-10
    B = 0.9
    mu_k = mu/np.sqrt(v+E)
    w  -= mu_k * gW
    v = B*v + (1-B)*np.square(gW)
    return(w,v)




def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)


def softmax_grad(x, y, w):
    #softmax
    z = np.dot(w,x)
    an = softmax(z)

    #Calculo del Costo
    divN = (-1/x.shape[1])
    tdotlogan = y * np.log(an)
    #lambdotW = (lambW/2) * np.linalg.norm(w,2)
    cost = divN * np.sum(np.sum(tdotlogan,axis=0, keepdims=True))
    Cost = cost # + lambdotW

    #Calculo del Gradiente
    error = (y-an)
    errordotX = np.dot(error,x.T)
    #lambadotW = lambW * w
    gradW = divN * errordotX # + lambadotW

    #gradW = -1 / N * ((T - A) * x.T)

    return gradW, Cost


def encoder(x, w):
    for i in range(int(len(w)/2)):
        x = act_sigmoid(np.dot(w[i], x))
    return x

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
# Configuration of the SNN
def load_config():      
    par = np.genfromtxt("./data/param_dae.csv",delimiter=',',dtype=None)    
    par_sae=[]    
    par_sae.append(np.float(par[0])) # Learn rate
    par_sae.append(np.float(par[1])) # miniBatchSize
    par_sae.append(np.int16(par[2])) # MaxIter
    for i in range(3,len(par)):
        par_sae.append(np.int16(par[i]))
    par    = np.genfromtxt("./data/param_softmax.csv",delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning     
    return(par_sae,par_sft)

# Load data 
def load_data_csv(fname):
    x     = pd.read_csv(fname, header = None)
    x     = np.array(x)  
    return(x)

def load_data_csv(fname):

    x = pd.read_csv(fname, header=None)
    x = np.array(x)

    return x


def save_w_dl(W, ws, nfile_w, cost, nfile_sft):
    W.append(ws)
    np.savez(nfile_w, idx=W, dtype=object)
    np.savetxt(nfile_sft, cost, delimiter=",", fmt="%.6f")


def load_w_dl(nfile):
    weigth = np.load(nfile, allow_pickle=True)
    w = weigth["idx"]
    return w[()]
