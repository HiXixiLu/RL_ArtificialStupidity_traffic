import copy
import numpy as np 
from numpy import linalg as la
import public_data as pdata
from RL_models.vehicle import motorVehicle, Bicycle


class IntersectionEnvironment():
    # class property —— 在整个实例化的对象中是公用的，且通常不作为实例变量使用
    rotate30_mat = np.array([[np.cos(np.pi/6), -1/2],[1/2, np.cos(np.pi/6)]])
    rotate90_mat = np.array([[0, -1], [1, 0]])
    theta = np.pi / 12
    ob_unit = 2     # 蛛网观察区域的单位腰长

    # 写死的12个合法行驶区域，对应12个 出发-目的 对
    polygons = {}

    # 写死的 4 个目的地区域
    # des_box = {}
    
    # 写死的 8 条边界
    edges = []

    def __init__(self, logger):       
        # 保存三种 agent 的集合 —— self.property 是一种实例属性
        self.vehicle_set = []       # TODO: 这里增加对 .enter_frame 进行排序的过程，以便计算时候能够拿到车辆运动先后顺序 —— 增加一个方法
        self.pedestrian_set = []
        self.logger = logger
        # 12个车辆合法行驶区域
        if len(self.polygons) == 0:
            # east to north
            self.polygons['e_n'] = np.array([
                [pdata.LANE_L+ 2 * pdata.LANE_W, pdata.LANE_W],
                [pdata.LANE_W, pdata.LANE_W],
                [pdata.LANE_W, pdata.LANE_L + 2 * pdata.LANE_W],    # 在道路仿真边缘区域增加一点用于合法区域判定
                [0, pdata.LANE_L + 2 * pdata.LANE_W],
                [0, 0],
                [pdata.LANE_L + 2 * pdata.LANE_W, 0] ])
            # east to west
            self.polygons['e_w'] = np.array([
                [pdata.LANE_L+2*pdata.LANE_W, pdata.LANE_W],
                [-pdata.LANE_L - 2 * pdata.LANE_W, pdata.LANE_W],
                [-pdata.LANE_L - 2 * pdata.LANE_W, 0],
                [pdata.LANE_L + 2 * pdata.LANE_W, 0]
            ])
            # east to south
            self.polygons['e_s'] = np.array([
                [pdata.LANE_L+2*pdata.LANE_W, pdata.LANE_W],
                [0, pdata.LANE_W],
                [-pdata.LANE_W, -pdata.LANE_W],
                [-pdata.LANE_W, -pdata.LANE_L - 2*pdata.LANE_W],
                [0, -pdata.LANE_L - 2 * pdata.LANE_W],
                [0, -pdata.LANE_W],
                [pdata.LANE_W, 0],
                [pdata.LANE_L + 2*pdata.LANE_W, 0]
            ])
            # north to west
            self.polygons['n_w'] = np.array([
                [-pdata.LANE_W, pdata.LANE_L + 2*pdata.LANE_W],
                [-pdata.LANE_W, pdata.LANE_W],
                [-pdata.LANE_L - 2 * pdata.LANE_W, pdata.LANE_W],
                [-pdata.LANE_L - 2 * pdata.LANE_W, 0],
                [0, 0],
                [0, pdata.LANE_L + 2*pdata.LANE_W]
            ])
            # north to south
            self.polygons['n_s'] = np.array([
                [-pdata.LANE_W, pdata.LANE_L + 2*pdata.LANE_W],
                [-pdata.LANE_W, -pdata.LANE_L - 2 * pdata.LANE_W],
                [0, -pdata.LANE_L - 2 * pdata.LANE_W],
                [0, pdata.LANE_L + 2*pdata.LANE_W]
            ])
            # north to east
            self.polygons['n_e'] = np.array([
                [-pdata.LANE_W, pdata.LANE_L + 2*pdata.LANE_W],
                [-pdata.LANE_W, 0],
                [pdata.LANE_W, -pdata.LANE_W],
                [pdata.LANE_L + 2*pdata.LANE_W, -pdata.LANE_W],
                [pdata.LANE_L + 2*pdata.LANE_W, 0],
                [pdata.LANE_W, 0],
                [0, pdata.LANE_W],
                [0, pdata.LANE_L + 2*pdata.LANE_W]
            ])
            # west to south
            self.polygons['w_s'] = np.array([
                [-pdata.LANE_L-2*pdata.LANE_W, -pdata.LANE_W],
                [-pdata.LANE_W, -pdata.LANE_W],
                [-pdata.LANE_W, -pdata.LANE_L - 2 * pdata.LANE_W],
                [0, -pdata.LANE_L - 2 * pdata.LANE_W],
                [0, 0]
            ])
            # west to east
            self.polygons['w_e'] = np.array([
                [-pdata.LANE_L-2*pdata.LANE_W, -pdata.LANE_W],
                [pdata.LANE_L + 2 * pdata.LANE_W, -pdata.LANE_W],
                [pdata.LANE_L + 2 * pdata.LANE_W, 0],
                [-pdata.LANE_L-2*pdata.LANE_W, 0]
            ])
            # west to north
            self.polygons['w_n'] = np.array([
                [-pdata.LANE_L-2*pdata.LANE_W, -pdata.LANE_W],
                [0, -pdata.LANE_W],
                [pdata.LANE_W, pdata.LANE_W],
                [pdata.LANE_W, pdata.LANE_L + 2*pdata.LANE_W],
                [0, pdata.LANE_L + 2*pdata.LANE_W],
                [0, pdata.LANE_W],
                [-pdata.LANE_W, 0],
                [-pdata.LANE_L - 2*pdata.LANE_W, 0]
            ])
            # south to east
            self.polygons['s_e'] = np.array([
                [pdata.LANE_W, -pdata.LANE_L - 2*pdata.LANE_W],
                [pdata.LANE_W, -pdata.LANE_W],
                [pdata.LANE_L + 2 * pdata.LANE_W, -pdata.LANE_W],
                [pdata.LANE_L + 2 * pdata.LANE_W, 0],
                [0, 0],
                [0, -pdata.LANE_L - pdata.LANE_W]
            ])
            # south to north
            self.polygons['s_n'] = np.array([
                [pdata.LANE_W, -pdata.LANE_L - 2*pdata.LANE_W],
                [pdata.LANE_W, pdata.LANE_L + 2 * pdata.LANE_W],
                [0, pdata.LANE_L + 2 * pdata.LANE_W],
                [0, -pdata.LANE_L - 2*pdata.LANE_W]
            ])
            # south to west
            self.polygons['s_w'] = np.array([
                [pdata.LANE_W, -pdata.LANE_L - 2*pdata.LANE_W],
                [pdata.LANE_W, 0],
                [0, pdata.LANE_W],
                [-pdata.LANE_L - 2*pdata.LANE_W, pdata.LANE_W],
                [-pdata.LANE_L - 2*pdata.LANE_W, 0],
                [-pdata.LANE_W, 0],
                [0, -pdata.LANE_W],
                [0, -pdata.LANE_L - 2*pdata.LANE_W]
            ])

        # 初始化车辆目的地判定区域（Xixi 你是不是写碰撞写到失心疯）
        # if len(self.des_box) == 0:
        #     # north destination
        #     self.des_box['north'] = np.array([  # 防止越界开车，因此矩形被限制在合法道路区域内
        #         # 坐标点顺序：右下-右上-左上-左下
        #         [pdata.LANE_W, pdata.LANE_L + pdata.LANE_W],
        #         [pdata.LANE_W, 2 * pdata.LANE_L + pdata.LANE_W],
        #         [0, 2 * pdata.LANE_L + pdata.LANE_W],
        #         [0, pdata.LANE_L + pdata.LANE_W]
        #     ])
        #     # west destination
        #     self.des_box['west'] = np.array([
        #         [-pdata.LANE_L - pdata.LANE_W, 0],
        #         [-pdata.LANE_L - pdata.LANE_W, pdata.LANE_W],
        #         [- 2* pdata.LANE_L - pdata.LANE_W, pdata.LANE_W],
        #         [- 2* pdata.LANE_L - pdata.LANE_W, 0]
        #     ])
        #     # south
        #     self.des_box['south'] = np.array([
        #         [0, - 2* pdata.LANE_L - pdata.LANE_W],
        #         [0, -pdata.LANE_L - pdata.LANE_W],
        #         [-pdata.LANE_W, -pdata.LANE_L - pdata.LANE_W],
        #         [-pdata.LANE_W, - 2* pdata.LANE_L - pdata.LANE_W]
        #     ])
        #     # east
        #     self.des_box['east'] = np.array([
        #         [2*pdata.LANE_L + pdata.LANE_W, -pdata.LANE_W],
        #         [2*pdata.LANE_L + pdata.LANE_W, 0],
        #         [pdata.LANE_L + pdata.LANE_W, 0],
        #         [pdata.LANE_L + pdata.LANE_W, -pdata.LANE_W]
        #     ])

        if len(self.edges) == 0:
            seg = np.array([[-pdata.LANE_L - pdata.LANE_W, -pdata.LANE_W],[-pdata.LANE_W, -pdata.LANE_W]])
            self.edges.append(seg)
            seg = np.array([[-pdata.LANE_W, -pdata.LANE_W], [-pdata.LANE_W, -pdata.LANE_L - pdata.LANE_W]])
            self.edges.append(seg)
            seg = np.array([[pdata.LANE_W, -pdata.LANE_L - pdata.LANE_W], [pdata.LANE_W, -pdata.LANE_W]])
            self.edges.append(seg)
            seg = np.array([[pdata.LANE_W, -pdata.LANE_W], [pdata.LANE_L + pdata.LANE_W, -pdata.LANE_W]])
            self.edges.append(seg)
            seg = np.array([[pdata.LANE_L + pdata.LANE_W, pdata.LANE_W], [pdata.LANE_W, pdata.LANE_W]])
            self.edges.append(seg)
            seg = np.array([[pdata.LANE_W, pdata.LANE_W], [pdata.LANE_W, pdata.LANE_L + pdata.LANE_W]])
            self.edges.append(seg)
            seg = np.array([[-pdata.LANE_W, pdata.LANE_L + pdata.LANE_W], [-pdata.LANE_W, pdata.LANE_W]])
            self.edges.append(seg)
            seg = np.array([[-pdata.LANE_W, pdata.LANE_W], [-pdata.LANE_L - pdata.LANE_W, pdata.LANE_W]])
            self.edges.append(seg)

        print("Environment initiation's done.")


    def __del__(self):
        self.vehicle_set.clear()
        del self.vehicle_set
        self.pedestrian_set.clear()
        del self.pedestrian_set
        print("Diconstruction's done")


    def step(self, agent, action):
        self._update_environment(agent, action)
        next_state = self._get_state_feature(agent)
        reward = self._get_reward(agent)
        done = self._check_termination(reward)

        return next_state, reward, done

    def _update_environment(self, agent, action):
        # 更新各个agent的顶点信息、速度属性
        # 如果有渲染线程，还需要重新渲染
        agent.update_attr(action)


    # 扇形观察区的检测(2020-1-16: 为了简化计算改用三角形替代, 正12边形观察区域)
    def _get_state_feature(self, agent):
        state_feature = np.ones(pdata.STATE_DIMENSION)      # default dtype = float

        pos = agent.get_position()  # agent 的位置是观察区的圆心
        vec = agent.get_velocity()
        vec_unit = vec / la.norm(vec)     # vec方向的单位向量

        # 48 个区域的观察区 —— state_feature 的前 48 个维度装着通行鼓励值
        # TODO: 关于其他Agent运动速度矢量穿越的衰减采集还未能完成
        i, section_i = 0, 0   
        tmp_unit = copy.deepcopy(vec_unit)
        while i < 48:
            tri_vtx = []
            if i % 4 == 0:
                vec_unit = copy.deepcopy(tmp_unit)
                tmp_unit = np.matmul(self.rotate30_mat, vec_unit)
            section_i = i % 4 + 1

            vtx1 = pos + section_i * self.ob_unit * vec_unit
            vtx2 = pos + section_i * self.ob_unit * tmp_unit
            tri_vtx.append(pos)
            tri_vtx.append(vtx1)
            tri_vtx.append(vtx2)    # 三角区的三个顶点

            # 三角观察区有相交: 1)三角区与其他agent的相撞 2)三角区与道路边界的相撞
            if self._tri_rect_collision_test(tri_vtx, agent.get_vertice()) or self._tri_bound_collision_test(tri_vtx):
                state_feature[i] = pdata.O_MAX
                # 之后所有的三角形通行鼓励值都下降
                i += 1
                while (i % 4) != 0:
                    state_feature[i] = pdata.O_MAX 
                    i += 1
                i -= 1

            i += 1

        state_feature[48], state_feature[49] = vec[0], vec[1]
        state_feature[50], state_feature[51] = pos[0], pos[1]
        state_feature[52] = la.norm(pos - agent.get_destination())

        # if isinstance(agent, MotorVehicle):
        #     print('Hello agent MotorVehicle.')

        # elif isinstance(agent, NonMotorVehicle):
        #     print('...')

        # elif isinstance(agent, Pedestrian):
        #     print('...')

        # else:
        #     print("Parameter agent has to be one of class MotorVehicle, NonMotorVehicle and Pedestrian.")

        return state_feature


    # when collision happened, return true
    # 需要逆时针排列的顶点坐标: tri_vtx.shape=(3,2), rect_vtx.shape=(4,2)
    # 先进行三边相交测试，再进行矩形顶点是否在三角形内测试
    def _tri_rect_collision_test(self, tri_vtx, rect_vtx):
        intersected = False
        # 1) 线段与矩形是否相交的测试 —— 若线段与多边形所有的边都没有交点，则不相交（可能在内部的情况不用考虑了，因为另两条边一定会相交）
        for i in range(0, len(tri_vtx)):
            seg = np.array([tri_vtx[i], tri_vtx[(i+1)%3]])
            if self._segment_test(seg, rect_vtx):
                return True

        # 2) 顶点是否在三角形内，使用逆时针连接的三边做向量点积就可以了（因为没有可能出现三角形在矩形内部的情况）
        intersected = self._vertex_test(tri_vtx, rect_vtx)
        return intersected

   
    # 检测观察区三角与马路边界的交叉
    def _tri_bound_collision_test(self, tri_vtx):
        # tri_vtx: a np.ndarray with shape of (3,2)
        for i in range(0, len(tri_vtx)):
            seg = np.array([tri_vtx[i], tri_vtx[(i+1)%3]])
            for j in range(0, 8):
                if self._seg_seg_test(seg, self.edges[j]):
                    return True
        return False


    # 判断两个线段是否相交：相交则返回True，否则返回False
    def _seg_seg_test(self, seg1, seg2):
        # seg1, seg2 : np.ndarray
        # 快速排除测试
        if not self.judge_aabb(seg1, seg2):
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
    def judge_aabb(self, seg1, seg2):
        return (min(seg1[0][0], seg1[1][0]) <= max(seg2[0][0], seg2[1][0]) and 
        max(seg1[0][0], seg1[1][0]) >= min(seg2[0][0], seg2[1][0]) and
        min(seg1[0][1], seg1[1][1]) <= max(seg2[0][1], seg2[1][1]) and
        max(seg1[0][1], seg1[1][1]) >= min(seg2[0][1], seg2[1][1]))


    # 注意参数这里是传引用
    # seg.shape = (2,2), rect_vtx.shape = (4, 2)
    def _segment_test(self, seg, rect_vtx):
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
        u_b_y = np.matmul(self.rotate90_mat, u_b_x)
        base_b = np.array([u_b_x, u_b_y])  # 基底
        base_b_reverse = la.inv(base_b)

        # segment坐标系相对于原点的平移
        translation_mat = np.array([[1, 0, seg_copy[0][0]], [0, 1, seg_copy[0][1]], [0, 0, 1]])
        # 矩形顶点坐标（原点系）到segment坐标系的变换
        for i in range(0, len(rect_vtx_copy)):
            tmp = np.ndarray.tolist(rect_vtx_copy[i])
            tmp.append(1)
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


    def _vertex_test(self, tri_vtx, rect_vtx):
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


    # shaped reward: 需要专家知识
    def _get_reward(self, agent, goal=None):
        reward = 0

        # 主线奖励：进入目的地、进入与目的地不符的区域、发生车辆行人碰撞，都会采用主线奖励
        if self._check_agent_collision(agent):
            reward = -5000
            self.logger.write_to_log('<agent collided>')
            return reward
        # elif not self._check_bound(agent):  # 包括越出仿真区域、开到左车道都会直接终止
        #     reward = -5000
        #     self.logger.write_to_log('<agent out of bound>')
        #     return reward
        elif self._check_arrival(agent):
            reward = 5000
            self.logger.write_to_log('<agent arrived>')
            return reward
        elif self._check_outside_region(agent):
            reward = -5000
            self.logger.write_to_log('<agent out>')
            return reward

        # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
        delta_d = la.norm(agent.get_last_position() - agent.get_destination()) - la.norm(agent.get_position() - agent.get_destination())
        _r_d = 300 * delta_d

        # 保持车道奖励
        if self._check_bound(agent):
            _r_lane_change = 300
        else:
            _r_lane_change = -300

        # 速度奖励
        velocity = agent.get_velocity()
        if la.norm(velocity) >= pdata.VELOCITY_LIMIT:
            # _r_v = -300  
            _r_v = -600
        else:
            # cos奖励 —— 与初始速度越接近，奖励越大
            origin_v = agent.get_origin_v()
            cosine = origin_v.dot(velocity) / (la.norm(origin_v) * la.norm(velocity)) 
            _r_v = 100 * cosine + 10 * la.norm(velocity)                 

        # if isinstance(agent, MotorVehicle):     # 包导入路径的写法都会影响 isinstance()的准确性
        #     print('Hello agent MotorVehicle.')

        # elif isinstance(agent, NonMotorVehicle):
        #     print('...')

        # elif isinstance(agent, Pedestrian):
        #     print('...')

        # else:
        #     print("Parameter agent has to be one of class MotorVehicle, NonMotorVehicle and Pedestrian.")

        reward = _r_d + _r_v + _r_lane_change
        # print('reward: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity), file = pdata.EXPERIMENT_LOG)
        self.logger.write_to_log('reward: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity))
        return reward 

    # 除了移动目标不一致，其余与
    def get_her_reward(self, agent, normal_reward, future_pos):
        if abs(normal_reward) == 5000:
            reward = normal_reward
        else:
            # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
            delta_d = la.norm(agent.get_last_position() - future_pos)
            _r_d = 300 * delta_d

            # 保持车道奖励
            if self._check_bound(agent):
                _r_lane_change = 300
            else:
                _r_lane_change = -300

            # 速度奖励
            velocity = agent.get_velocity()
            if la.norm(velocity) >= pdata.VELOCITY_LIMIT:
                # _r_v = -300  
                _r_v = -600
            else:
                # cos奖励 —— 与初始速度越接近，奖励越大
                origin_v = agent.get_origin_v()
                cosine = origin_v.dot(velocity) / (la.norm(origin_v) * la.norm(velocity)) 
                _r_v = 100 * cosine + 10 * la.norm(velocity)      

            reward  =_r_d + _r_lane_change + _r_v
        return reward


    # 检查是否终止
    def _check_termination(self, reward):
        if abs(reward) == 5000:
            return True
        else:
            return False
        

    # Agent间的碰撞检查 —— SAT算法
    # TODO: 针对多智能体碰撞的检测
    def _check_agent_collision(self, agent):
        collision = False
        _count = 0

        # 姑且采用暴力遍历方式检查相撞
        if isinstance(agent, motorVehicle):
            _count = len(self.motor_set)
            for i in range(0, _count):
                if agent == self.motor_set[i]:
                    continue
                elif self._check_bilateral_collision(agent, self.motor_set):
                    collision = True
            collision = False            

        # elif isinstance(agent, NonMotorVehicle):
        #     print('...')

        # elif isinstance(agent, Pedestrian):
        #     print('...')

        else:
            print("Parameter agent has to be one of class MotorVehicle, NonMotorVehicle and Pedestrian.")

        return collision


    # 矩形间的碰撞检测 —— 由于存在斜交的可能性，直接比较线段
    # TODO: 暴力遍历检查该怎么优化？
    def _check_bilateral_collision(self, agent1, agent_set):
        vertice1 = agent1.get_vertice()
        for i in range(0, len(agent_set)):
            vertice2 = agent_set[i].get_vertice()
            if self._check_obb_collision(vertice1, vertice2):
                return True
        return False
        

    # OBB盒碰撞测试
    def _check_obb_collision(self, box1, box2):
        #return Ture: if obb has collision
        if box1.shape != (4,2) or box2.shape != (4,2):
            print("_check_obb_collision : invalid argument")
            return False
        for i in range(0, 4):
            seg = np.array([box1[i], box1[(i+1)%4]])
            if self._segment_test(seg, box2):
                return True
        return False


    # 矩形是否行驶在合法区域内的碰撞检测 —— 顶点是否全部在合理行驶区域内
    def _check_bound(self, agent):
        # return False : if box is not inside the bound
        vertice = agent.get_vertice()
        _origin = agent.get_origin()
        _des = agent.get_des_string()
        if _origin == 'east' and _des == 'south':
            return self.box_inside_polygon(vertice, self.polygons['e_s'])
        elif _origin == 'east' and _des == 'west':
            return self.box_inside_polygon(vertice, self.polygons['e_w'])
        elif _origin == 'east' and _des == 'north':
            return self.box_inside_polygon(vertice, self.polygons['e_n'])  
        elif _origin == 'north' and _des == 'east':
            return self.box_inside_polygon(vertice, self.polygons['n_e'])
        elif _origin == 'north' and _des == 'south':
            return self.box_inside_polygon(vertice, self.polygons['n_s'])
        elif _origin == 'north' and _des == 'west':
            return self.box_inside_polygon(vertice, self.polygons['n_w'])
        elif _origin == 'west' and _des == 'south':
            return self.box_inside_polygon(vertice, self.polygons['w_s'])
        elif _origin == 'west' and _des == 'east':
            return self.box_inside_polygon(vertice, self.polygons['w_e'])
        elif _origin == 'west' and _des == 'north':
            return self.box_inside_polygon(vertice, self.polygons['w_n'])
        elif _origin == 'south' and _des == 'east':
            return self.box_inside_polygon(vertice, self.polygons['s_e'])
        elif _origin == 'south' and _des == 'north':
            return self.box_inside_polygon(vertice, self.polygons['s_n'])
        elif _origin == 'south' and _des == 'west':
            return self.box_inside_polygon(vertice, self.polygons['s_w'])


    # 是否到达目标点的测量 —— 以是否与路口大矩形相交来判断（Xixi 你是不是写碰撞写到失心疯）
    # def _check_arrival(self, agent):
    #     # return True: if agent arrived their destination.
    #     arrival = False
    #     agent_box = agent.get_vertice()
    #     # 2020-2-6：进入目标矩形就算到达目的地
    #     des = agent.get_des_string()
    #     if des == 'north':
    #         arrival = self._check_obb_collision(agent_box, self.des_box['north'])
    #     elif des == 'west':
    #         arrival = self._check_obb_collision(agent_box, self.des_box['west'])
    #     elif des == 'south':
    #         arrival = self._check_obb_collision(agent_box, self.des_box['south'])
    #     elif des == 'east':
    #         arrival = self._check_obb_collision(agent_box, self.des_box['east'])
    #     return arrival

    # 是否到达目标点 —— 用距离来判断
    def _check_arrival(self, agent):
        # return True: if agent arrived their destination
        distance = la.norm(agent.get_position() - agent.get_destination())
        if distance <= (agent.get_size_width()/2):
            return True
        else:
            return False


    # return True when agent's position is out of region
    def _check_outside_region(self, agent):
        pos = agent.get_position()
        if pos[0] <= -pdata.LANE_W and pos[1] <= -pdata.LANE_W:
            return True
        elif pos[0] >= pdata.LANE_W and pos[1] <= -pdata.LANE_W:
            return True
        elif pos[0] >= pdata.LANE_W and pos[1] >= pdata.LANE_W:
            return True
        elif pos[0] <= -pdata.LANE_W and pos[1] >= pdata.LANE_W:
            return True
        elif pos[0] > (pdata.LANE_W + pdata.LANE_L) or pos[0] < (-pdata.LANE_L - pdata.LANE_W):
            return True
        elif pos[1] > (pdata.LANE_W + pdata.LANE_L) or pos[1] < (-pdata.LANE_L - pdata.LANE_W):
            return True
        return False

    
    # 返回单个步长内需要渲染的所有对象的顶点信息
    def get_render_info(self):
        info_list = []
        # TODO: 后期加入非机动车、行人的更新代码
        for i in range(0, len(self.motor_set)):
            vertice = self.motor_set[i].get_vertice()
            info_list.append(vertice)   # 不能直接往numpy数组里添加元素，要先使用python原生list

        info_list = np.array(info_list)  
        return info_list      


    # 点在多边形内部的射线检测
    # point : 表示平面点坐标的 numpy ndarray
    # polygon：表示平面多边形的顶点集合
    def rt_test(self, point, polygon): 
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
                slope = (_start[1] - _end[1]) / (_start[0] - _end[1])
                cross_x = (radio_y - _start[1]) / slope + _start[0]
                if cross_x < point[0]:
                    left_count += 1
                elif cross_x > point[1]:
                    right_count += 1

        if left_count & 1 == 1 and right_count  & 1 == 1:
            _inside = True

        return _inside


    # 矩形是否在某个多边形内部的检测
    def box_inside_polygon(self, box, polygon):
        if not isinstance(box, np.ndarray) or not isinstance(polygon, np.ndarray):
            print('box_inside_polygon : arguments should be a numpy ndarray')

        count = box.shape[0]
        for i in range(count):
            if not self.rt_test(box[i], polygon):
                return False

        return True


    # 环境的 reset —— 重置单个agent
    def reset(self,agent):
        # print('\n -----------agent reset---------- \n')
        self.logger.write_to_log('\n -----------agent reset---------- \n')
        agent.reset_agent()
        return self._get_state_feature(agent)