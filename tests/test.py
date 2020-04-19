import numpy as np
from itertools import count
import sys, os, time, math, multiprocessing,copy
sys.path.append(os.getcwd() + '/traffic_model/RL_models')  
from numpy import linalg as la
import torch
import torch.nn.functional as F


from threading import Timer

def fun():
    print("hello, world")


if __name__=='__main__':
    n = np.array([1,2,3,4])
    a = np.mean(n).item()
    print(a, type(a))

