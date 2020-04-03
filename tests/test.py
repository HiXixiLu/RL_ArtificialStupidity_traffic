import numpy as np
from itertools import count
import sys, os, time, math, multiprocessing,copy
sys.path.append(os.getcwd() + '/traffic_model/RL_models')  
from numpy import linalg as la


     
if __name__ == '__main__':
    n1 = np.array([1,2,3])
    n2 = np.array([2,3,4])
    li = []
    li.append(n1)
    li.append(n2)
    print(li)