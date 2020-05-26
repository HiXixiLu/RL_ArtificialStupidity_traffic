import numpy as np
import numpy.linalg as la
import sys, os, time, math, multiprocessing,copy
sys.path.append(os.getcwd() + '/traffic_model/RL_models')  
import torch as th 



if __name__=='__main__':
    a = np.array([1,2,3])
    b = th.FloatTensor(a)
    print(b.shape)
    b = b[:, None]
    print(b, b.shape)