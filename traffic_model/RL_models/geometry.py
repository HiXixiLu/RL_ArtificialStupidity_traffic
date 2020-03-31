''' 一个几何数据模块，类似于 C++ 中的 struct '''
import numpy as np
import numpy.linalg as la


rotate90_mat = np.array([[0, -1],[1, 0]])
rotate30_mat = np.array([[np.cos(np.pi/6), -1/2],[1/2, np.cos(np.pi/6)]])

class ray():
    s_point = np.ones((2))
    cos = 0.0
    sin = 0.0
     
    def __init__(self, start, cosin, sin):
        self.s_point[0], self.s_point[1] = start[0], start[1]
        self.cos = cosin
        self.sin = sin


def copy_ray(a_ray):
    new_one = ray(a_ray.s_point, a_ray.cosin, a_ray.sin)
    return new_one

def get_rotation_thirty_ray(a_ray):
    vec = np.array([a_ray.cos, a_ray.sin])
    vec = np.matmul(rotate30_mat, vec)
    new_one = ray(a_ray.s_point, vec[0], vec[1])
    return new_one