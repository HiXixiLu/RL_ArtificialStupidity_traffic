''' 几何计算模块 '''
import copy
import numpy as np
import numpy.linalg as la
import public_data as pdata


rotate90_mat = np.array([[0.0, -1.0],[1.0, 0.0]])
rotate30_mat = np.array([[np.cos(np.pi/6), -1/2],[1/2, np.cos(np.pi/6)]])
rotate_ng_90_mat = np.array([[0.0, 1.0], [-1.0, 0.0]])

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


# 直线方程： ax + by = c
def get_ray_box_crosspoints(ray, box_vertices):
    points = []
    for i in range(0, len(box_vertices)):
        seg = [box_vertices[i], box_vertices[(i+1)%len(box_vertices)]]
        if cross_check(seg, ray):
            seg_a = -(seg[1][1]-seg[0][1]) / (seg[1][0] - seg[0][0] if seg[1][0] - seg[0][0] else pdata.EPSILON) # -k:k为斜率
            seg_b = 1.0
            seg_c = seg_a * seg[0][0] + seg[0][1]
            # 直线交点
            point = la.solve(np.array([[-ray.sin, ray.cos], [seg_a, seg_b]]), np.array([-ray.sin*ray.s_point[0]+ray.cos*ray.s_point[1], seg_c]))
            # 是否在射线上
            tmp_vec = point - ray.s_point
            dot_product = tmp_vec.dot(np.array([ray.cos, ray.sin]))
            if dot_product > pdata.EPSILON:
                points.append(copy.deepcopy(point))
    return points


# ax + by = c
def get_seg_ray_crosspoints(ray, seg_list):
    points = []
    for i in range(0, len(seg_list)):
        seg = seg_list[i]
        if cross_check(seg, ray):
            seg_a = -(seg[1][1]-seg[0][1]) / (seg[1][0] - seg[0][0] if seg[1][0] - seg[0][0] else pdata.EPSILON) # -k:k为斜率
            seg_b = 1.0
            seg_c = seg_a * seg[0][0] + seg[0][1]
            # 直线交点
            point = la.solve(np.array([[-ray.sin, ray.cos], [seg_a, seg_b]]), np.array([-ray.sin*ray.s_point[0]+ray.cos*ray.s_point[1], seg_c]))
            # 是否在射线上
            tmp_vec = point - ray.s_point
            dot_product = tmp_vec.dot(np.array([ray.cos, ray.sin]))
            if dot_product > pdata.EPSILON:
                points.append(copy.deepcopy(point))
    return points  


# 端点异侧返回 True，否则返回 False
def cross_check(seg, ray):
    vec1, vec2 = seg[0] - ray.s_point, seg[1] - ray.s_point
    unit = np.array([ray.cos, ray.sin])
    res = np.cross(unit, vec1) * np.cross(unit, vec2)
    return res < 0


# 判断两个线段是否相交：相交则返回True，否则返回False
def seg_seg_test(seg1, seg2):
    # seg1, seg2 : np.ndarray
    # 快速排除测试
    if not judge_aabb(seg1, seg2):
        return False
    # 求叉乘
    sign = 0
    vec = seg1[1] - seg1[0]
    tmp_vec1, tmp_vec2 = seg2[0] - seg1[0], seg2[1] - seg1[0]
    cross1 = np.cross(vec, tmp_vec1)
    cross2 = np.cross(vec, tmp_vec2)
    if cross1 * cross2 <= 0:
        sign += 1
    vec = seg2[1] - seg2[0]
    tmp_vec1, tmp_vec2 = seg1[0] - seg2[0], seg1[1] - seg2[0]
    cross1 = np.cross(vec, tmp_vec1)
    cross2 = np.cross(vec, tmp_vec2)
    if cross1 * cross2 <= 0:
        sign += 1
    if sign == 2:
        return True
    else:
        return False   


# 测试 AABB盒 是否直接分离，如果相交返回True，否则返回False
def judge_aabb(seg1, seg2):
    return (min(seg1[0][0], seg1[1][0]) <= max(seg2[0][0], seg2[1][0]) and 
    max(seg1[0][0], seg1[1][0]) >= min(seg2[0][0], seg2[1][0]) and
    min(seg1[0][1], seg1[1][1]) <= max(seg2[0][1], seg2[1][1]) and
    max(seg1[0][1], seg1[1][1]) >= min(seg2[0][1], seg2[1][1]))


# 注意参数这里是传引用
# seg.shape = (2,2), rect_vtx.shape = (4, 2)
def segment_test(seg, rect_vtx):
    flag = False
    delta = seg[0]  # delta:待移动的向量距离[x, y], 线段起点
    seg_copy, rect_vtx_copy = seg, rect_vtx

    # 平移
    for i in range(0, len(seg_copy)):
        seg_copy[i] -= delta
    for i in range(0, len(rect_vtx_copy)):
        rect_vtx_copy[i] -= delta

    # 计算以segment为横轴的坐标系的非平移的基在原点坐标系中的表示
    u_b_x = (seg_copy[1] - seg_copy[0]) / la.norm((seg_copy[1] - seg_copy[0]))  # u_x, u_y: 由线段确定的坐标系在原点坐标系中的基
    u_b_y = np.matmul(rotate90_mat, u_b_x)
    base_b = np.array([u_b_x, u_b_y])  # 基底
    base_b_reverse = la.inv(base_b)

    # segment坐标系相对于原点的平移
    translation_mat = np.array([[1, 0, seg_copy[0][0]], [0, 1, seg_copy[0][1]], [0, 0, 1]])
    # 矩形顶点坐标（原点系）到segment坐标系的变换
    for i in range(0, len(rect_vtx_copy)):
        tmp = np.ndarray.tolist(rect_vtx_copy[i])
        tmp.append(1            )
        tmp = np.array(tmp)   # [x, y, 1]
        tmp = np.matmul(translation_mat, tmp)
        tmp = tmp[0:2]  
        rect_vtx_copy[i] = np.matmul(base_b_reverse, tmp)   # 得到 segment 坐标系下的坐标值

    for i in range(0, len(rect_vtx_copy)):
        seg_line = np.array([rect_vtx_copy[i], rect_vtx_copy[(i+1)%4]])
        if seg_line[0][1] * seg_line[1][1] > 0:
            continue    # 不相交
        else:
            x1, y1, x2, y2 = seg_line[0][0], seg_line[0][1], seg_line[1][0], seg_line[1][1]
            x_p = y1 / (y2 - y1) * (x2 - x1) + x1   # 这里容易产生 Nan值
            if x_p > 0 and x_p < seg_copy[1][1]:
                flag = True  # 线段相交
                return flag
    return flag


def vertex_test(tri_vtx, rect_vtx):
    flag = False
    for i in range(0, len(rect_vtx)):
        vtx = rect_vtx[i]
        # 矩形顶点是否在三角形内
        count = 0
        for j in range(0, len(tri_vtx)):
            start = tri_vtx[j]
            end = tri_vtx[(j+1)%3]
            res = np.dot(end-start, vtx-start)
            # 当前点不在三角形内
            if res < 0:
                break
            else:
                count += 1
        # 只要有一个矩形顶点在三角形内
        if count >= 3:
            flag = True
            break
            
    return flag


 # OBB盒碰撞测试
def check_obb_collision(box1, box2):
    #return Ture: if obb has collision
    if box1.shape != (4,2) or box2.shape != (4,2):
        print("_check_obb_collision : invalid argument")
        return False
    for i in range(0, 4):
        seg = np.array([box1[i], box1[(i+1)%4]])
        if segment_test(seg, box2):
            return True
    return False


# 矩形是否在某个多边形内部的检测
def box_inside_polygon(box, polygon):
    if not isinstance(box, np.ndarray) or not isinstance(polygon, np.ndarray):
        print('box_inside_polygon : arguments should be a numpy ndarray')

    count = box.shape[0]
    for i in range(count):
        if not rt_test(box[i], polygon):
            return False

    return True


# 点在多边形内部的射线检测
# point : 表示平面点坐标的 numpy ndarray
# polygon：表示平面多边形的顶点集合
def rt_test(point, polygon): 
    _inside = False
    # 判断多边形的顶点是否有效
    if not isinstance(polygon, np.ndarray) or polygon.shape[1] != 2:
        print("rt_test : arguments should be a numpy ndarray")
        return

    if not isinstance(point, np.ndarray) or point.shape[0] != 2:
        print('rt_test : arguments should be a numpy ndarray')
        return

    _rows = polygon.shape[0]
    for i in range(_rows):
        _start, _end = polygon[i], polygon[(i+1)%_rows] 
        radio_y = point[1]      # 以x轴的平行线作为射线
        left_count , right_count = 0, 0

        if((_start[1] - radio_y) * (_end[1] - radio_y) < 0):
            slope = (_start[1] - _end[1]) / ((_start[0] - _end[1]) + pdata.EPSILON)
            cross_x = (radio_y - _start[1]) / slope + _start[0]
            if cross_x < point[0]:
                left_count += 1
            elif cross_x > point[1]:
                right_count += 1

    if left_count & 1 == 1 and right_count  & 1 == 1:
        _inside = True

    return _inside


def get_nearest_distance(origin, pos_list):
    nearest = float('inf')
    for i in range(0, len(pos_list)):
        distance = la.norm(pos_list[i] - origin)
        if distance < nearest:
            nearest = distance
    return nearest


# 标准世界坐标系为标准二维平面
def local_to_world(origin_pos, axis_y, local_coordinate):
    u_y = axis_y / (la.norm(axis_y) + pdata.EPSILON)
    u_x = np.matmul(rotate_ng_90_mat, u_y)
    transform_mat = np.array([[u_x[0], u_y[0], origin_pos[0]], [u_x[1],u_y[1], origin_pos[1]], [0.0, 0.0, 1.0]])
    co = list(local_coordinate)
    co.append(1.0)
    co = np.array(co)
    local_coordinate = np.matmul(transform_mat, co)
    return local_coordinate[:-1]

def world_to_local(origin_pos, axis_y, coordinate):
    u_y = axis_y / (la.norm(axis_y) + pdata.EPSILON)
    u_x = np.matmul(rotate_ng_90_mat, u_y)
    transform_mat = np.array([[u_x[0], u_y[0], origin_pos[0]], [u_x[1],u_y[1], origin_pos[1]], [0.0, 0.0, 1.0]])
    transform_mat = la.inv(transform_mat)
    co = list(coordinate)
    co.append(1.0)
    co = np.array(co)
    world_coordinate = np.matmul(transform_mat, co)
    return world_coordinate[:-1]
