import numpy as np
from itertools import count
import sys, os, time, math, multiprocessing,copy
sys.path.append(os.getcwd() + '/traffic_model/RL_models')  
from numpy import linalg as la
import torch
import torch.nn.functional as F


     
if __name__ == '__main__':
    print(torch.cuda.is_available())
