import numpy as np
from itertools import count
import sys, os, time
sys.path.append('D:\\python_project\\TrafficModel\\traffic_model\\RL_models')
import matplotlib.pyplot as plt
from numpy import linalg as la
import threading



class Pedestrian():
    priority = 0
    def __init__(self, index):
        self.priority = index


if __name__ == '__main__':
    a = 1.0
    b = 1
    print(type(a), type(b))
    if isinstance(a, float):
        print("YaaHoo")
    if isinstance(b, int):
        print("Bingo")