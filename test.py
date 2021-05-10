import pandas as pd
import numpy as np
import my_utility as ut

  
# Feed-forward of the DL
def forward_dl(xv,w):
	z = np.dot(w[str(0)],xv)
	for i in range(1,len(w)):
		a = ut.act_sigmoid(z)
		z = np.dot(w[str(i)],a)
	an = ut.softmax(z)
	return(an)

# Beginning ...
def main():		
	xv     = ut.load_data_csv('./data/test_x.csv')	
	yv     = ut.load_data_csv('./data/test_y.csv')
	#yv2     = ut.load_data_csv('./data/test_y2.csv')
	W      = ut.load_w_dl('./data/w_dl.npz')
	zv     = forward_dl(xv,W)      		
	ut.metricas(yv,zv)

if __name__ == '__main__':   
	 main()

