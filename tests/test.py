import numpy as np
from itertools import count
import sys
sys.path.append('D:\\python_project\\TrafficModel\\traffic_model\\RL_models')
import matplotlib.pyplot as plt
from numpy import linalg as la
import os
import public_data as pdata
import copy
import time
import torch


 # 测试 AABB盒 是否直接分离，如果相交返回True，否则返回False
def judge_aabb(seg1, seg2):
    return (min(seg1[0][0], seg1[1][0]) <= max(seg2[0][0], seg2[1][0]) and 
    max(seg1[0][0], seg1[1][0]) >= min(seg2[0][0], seg2[1][0]) and
    min(seg1[0][1], seg1[1][1]) <= max(seg2[0][1], seg2[1][1]) and
    max(seg1[0][1], seg1[1][1]) >= min(seg2[0][1], seg2[1][1]))


if __name__ == '__main__':
    a = np.array([1,2])
    b = np.array([2,3])
    c = a.dot(b)
    print(type(c), c)
    c = la.norm(a)
    print(type(c), c)