import numpy as np
from itertools import count
import sys, os, time
sys.path.append('D:\\python_project\\TrafficModel\\traffic_model\\RL_models')
import matplotlib.pyplot as plt
from numpy import linalg as la
import threading

exitFlag = 0

class A():
    def __init__(self):
        print('class A')


class B(A):
    def __init__(self):
        print('class B')


a = np.zeros(2)
b = np.ones(2)
combine = np.concatenate([a, b], axis = 0)
print(combine)
c = combine.reshape((2,2))
print(combine)
print(c)