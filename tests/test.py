import numpy as np
from itertools import count
import sys, os, time, math, multiprocessing
sys.path.append(os.getcwd() + '/traffic_model/RL_models')  
from numpy import linalg as la


def write_file(filename,num):
    target = open(filename, 'w')
    for i in range(1,num+1):
        target.write("%d line\n" % i)
     
if __name__ == '__main__':
    start = time.time()
     
    p1 = multiprocessing.Process(target=write_file,args=(os.getcwd() + '/traffic_model/data/'+'1.txt', 10))
    p2 = multiprocessing.Process(target=write_file,args=(os.getcwd() + '/traffic_model/data/'+'2.txt', 20))
     
    #启动子进程
    p1.start()
    p2.start()
     
    #等待fork的子进程终止再继续往下执行，可选填一个timeout参数
    p1.join()
    p2.join()
     
    end = time.time()
    print(str(round(end-start,3))+'s')