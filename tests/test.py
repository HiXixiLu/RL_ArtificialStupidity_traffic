import numpy as np
from itertools import count
import sys, os, time, math, multiprocessing,copy
sys.path.append(os.getcwd() + '/traffic_model/RL_models')  
from numpy import linalg as la
import torch
import torch.nn.functional as F


     
if __name__ == '__main__':
    n1 = np.array([1,1,3])
    n1 = n1.reshape(-1, 1)
    t = torch.FloatTensor(n1)
    t1 = torch.FloatTensor([1.0])
    res = F.mse_loss(t, t1)
    print(res)