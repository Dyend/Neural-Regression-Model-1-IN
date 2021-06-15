
import pandas as pd
import numpy as np
import my_utility as ut

   
# Feed-forward of the DL
def forward_dl(x,w):
    # completar
    
    return()  

# Beginning ...
def main():		
	xv     = ut.load_data_csv('test_x.csv')		
	yv     = ut.load_data_csv('test_y.csv')
	W      = ut.load_w_dl('w_dl.npz')
	zv     = forward_dl(xv,W)      		
	Fsc    = ut.metricas(yv,zv) 
	
if __name__ == '__main__':   
	 main()

