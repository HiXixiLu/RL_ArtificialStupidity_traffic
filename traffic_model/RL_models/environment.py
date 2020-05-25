import math
import copy
import numpy as np 
from numpy import linalg as la
import public_data as pdata
import geometry as geo
from RL_models.vehicle import motorVehicle, Bicycle, vehicle,MotorVehicleMA,BicycleMA
from RL_models.pedestrian import pedestrian, PedestrianMA


# 写死的12个合法行驶区域，对应12个 出发-目的 对
polygons = {}
polygons['e_n'] = np.array([
            [pdata.LANE_L+ 2 * pdata.LANE_W, pdata.LANE_W],
            [pdata.LANE_W, pdata.LANE_W],
            [pdata.LANE_W, pdata.LANE_L + 2 * pdata.LANE_W],    # 在道路仿真边缘区域增加一点用于合法区域判定
            [0, pdata.LANE_L + 2 * pdata.LANE_W],
            [0, 0],
            [pdata.LANE_L + 2 * pdata.LANE_W, 0] ])
# east to west
polygons['e_w'] = np.array([
    [pdata.LANE_L+2*pdata.LANE_W, pdata.LANE_W],
    [-pdata.LANE_L - 2 * pdata.LANE_W, pdata.LANE_W],
    [-pdata.LANE_L - 2 * pdata.LANE_W, 0],
    [pdata.LANE_L + 2 * pdata.LANE_W, 0]
])
# east to south
polygons['e_s'] = np.array([
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
polygons['n_w'] = np.array([
    [-pdata.LANE_W, pdata.LANE_L + 2*pdata.LANE_W],
    [-pdata.LANE_W, pdata.LANE_W],
    [-pdata.LANE_L - 2 * pdata.LANE_W, pdata.LANE_W],
    [-pdata.LANE_L - 2 * pdata.LANE_W, 0],
    [0, 0],
    [0, pdata.LANE_L + 2*pdata.LANE_W]
])
# north to south
polygons['n_s'] = np.array([
    [-pdata.LANE_W, pdata.LANE_L + 2*pdata.LANE_W],
    [-pdata.LANE_W, -pdata.LANE_L - 2 * pdata.LANE_W],
    [0, -pdata.LANE_L - 2 * pdata.LANE_W],
    [0, pdata.LANE_L + 2*pdata.LANE_W]
])
# north to east
polygons['n_e'] = np.array([
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
polygons['w_s'] = np.array([
    [-pdata.LANE_L-2*pdata.LANE_W, -pdata.LANE_W],
    [-pdata.LANE_W, -pdata.LANE_W],
    [-pdata.LANE_W, -pdata.LANE_L - 2 * pdata.LANE_W],
    [0, -pdata.LANE_L - 2 * pdata.LANE_W],
    [0, 0]
])
# west to east
polygons['w_e'] = np.array([
    [-pdata.LANE_L-2*pdata.LANE_W, -pdata.LANE_W],
    [pdata.LANE_L + 2 * pdata.LANE_W, -pdata.LANE_W],
    [pdata.LANE_L + 2 * pdata.LANE_W, 0],
    [-pdata.LANE_L-2*pdata.LANE_W, 0]
])
# west to north
polygons['w_n'] = np.array([
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
polygons['s_e'] = np.array([
    [pdata.LANE_W, -pdata.LANE_L - 2*pdata.LANE_W],
    [pdata.LANE_W, -pdata.LANE_W],
    [pdata.LANE_L + 2 * pdata.LANE_W, -pdata.LANE_W],
    [pdata.LANE_L + 2 * pdata.LANE_W, 0],
    [0, 0],
    [0, -pdata.LANE_L - pdata.LANE_W]
])
# south to north
polygons['s_n'] = np.array([
    [pdata.LANE_W, -pdata.LANE_L - 2*pdata.LANE_W],
    [pdata.LANE_W, pdata.LANE_L + 2 * pdata.LANE_W],
    [0, pdata.LANE_L + 2 * pdata.LANE_W],
    [0, -pdata.LANE_L - 2*pdata.LANE_W]
])
# south to west
polygons['s_w'] = np.array([
    [pdata.LANE_W, -pdata.LANE_L - 2*pdata.LANE_W],
    [pdata.LANE_W, 0],
    [0, pdata.LANE_W],
    [-pdata.LANE_L - 2*pdata.LANE_W, pdata.LANE_W],
    [-pdata.LANE_L - 2*pdata.LANE_W, 0],
    [-pdata.LANE_W, 0],
    [0, -pdata.LANE_W],
    [0, -pdata.LANE_L - 2*pdata.LANE_W]
])

# 写死的 4 个车道目的地终点线: south, east, north, west
des_seg = {}
des_seg['s'] = np.array([[-pdata.LANE_W, -pdata.LANE_L - pdata.LANE_W], [pdata.LANE_W,  -pdata.LANE_L - pdata.LANE_W]]) 
des_seg['e'] = np.array([[pdata.LANE_L+pdata.LANE_W, -pdata.LANE_W], [pdata.LANE_L+pdata.LANE_W, pdata.LANE_W]])
des_seg['n'] = np.array([[pdata.LANE_W, pdata.LANE_L + pdata.LANE_W], [0.0, pdata.LANE_L + pdata.LANE_W]])
des_seg['w'] = np.array([[-pdata.LANE_L - pdata.LANE_W, pdata.LANE_W], [-pdata.LANE_L-pdata.LANE_W, 0.0]])

# 写死的 8 条边界
edges = []
seg = np.array([[-pdata.LANE_L - pdata.LANE_W, -pdata.LANE_W],[-pdata.LANE_W, -pdata.LANE_W]])
edges.append(seg)
seg = np.array([[-pdata.LANE_W, -pdata.LANE_W], [-pdata.LANE_W, -pdata.LANE_L - pdata.LANE_W]])
edges.append(seg)
seg = np.array([[pdata.LANE_W, -pdata.LANE_L - pdata.LANE_W], [pdata.LANE_W, -pdata.LANE_W]])
edges.append(seg)
seg = np.array([[pdata.LANE_W, -pdata.LANE_W], [pdata.LANE_L + pdata.LANE_W, -pdata.LANE_W]])
edges.append(seg)
seg = np.array([[pdata.LANE_L + pdata.LANE_W, pdata.LANE_W], [pdata.LANE_W, pdata.LANE_W]])
edges.append(seg)
seg = np.array([[pdata.LANE_W, pdata.LANE_W], [pdata.LANE_W, pdata.LANE_L + pdata.LANE_W]])
edges.append(seg)
seg = np.array([[-pdata.LANE_W, pdata.LANE_L + pdata.LANE_W], [-pdata.LANE_W, pdata.LANE_W]])
edges.append(seg)
seg = np.array([[-pdata.LANE_W, pdata.LANE_W], [-pdata.LANE_L - pdata.LANE_W, pdata.LANE_W]])
edges.append(seg)

# 写死的4个路口目的地终点线
intersection_des_seg = {}
intersection_des_seg['s'] = np.array([[-pdata.LANE_W, -pdata.LANE_W], [0.0, -pdata.LANE_W]])
intersection_des_seg['e'] = np.array([[pdata.LANE_W, -pdata.LANE_W], [pdata.LANE_W, 0.0]])
intersection_des_seg['n'] = np.array([[pdata.LANE_W, pdata.LANE_W], [0.0, pdata.LANE_W]])
intersection_des_seg['w'] = np.array([[0.0, pdata.LANE_W], [-pdata.LANE_W, pdata.LANE_W]])


class TrainingEnvironment():
    # class property —— 在整个实例化的对象中是公用的，且通常不作为实例变量使用
    # rotate30_mat = np.array([[np.cos(np.pi/6), -1/2],[1/2, np.cos(np.pi/6)]])
    # rotate90_mat = np.array([[0, -1], [1, 0]])
    # theta = np.pi / 12
    # ob_unit = 2     # 蛛网观察区域的单位腰长

    def __init__(self, logger):       
        # 保存三种 agent 的集合 —— self.property 是一种实例属性
        self.vehicle_set = []       # TODO: 这里增加对 .enter_frame 进行排序的过程，以便计算时候能够拿到车辆运动先后顺序 —— 增加一个方法
        self.pedestrian_set = []
        self.logger = logger
            

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


    def __del__(self):
        self.vehicle_set.clear()
        del self.vehicle_set
        self.pedestrian_set.clear()
        del self.pedestrian_set


    def step(self, agent, action):
        self._update_environment(agent, action)
        next_state = self._get_state_feature(agent)
        if isinstance(agent, vehicle):
            reward = self._get_reward(agent)
        else:
            reward = self._get_reward_pe(agent)
        done = self._check_termination(reward)

        return next_state, reward, done


    # 该函数仅仅是训练时候用
    def _update_environment(self, agent, action):
        agent.update_attr(action)


    # 扇形观察区的检测(2020-1-16: 为了简化计算改用三角形替代, 正12边形观察区域)
    # def _get_state_feature(self, agent):
    #     state_feature = np.ones(pdata.STATE_DIMENSION)      # default dtype = float

    #     pos = agent.get_position()  # agent 的位置是观察区的圆心
    #     vec = agent.get_velocity()
    #     vec_unit = vec / la.norm(vec)     # vec方向的单位向量

    #     # 48 个区域的观察区 —— state_feature 的前 48 个维度装着通行鼓励值
    #     # TODO: 关于其他Agent运动速度矢量穿越的衰减采集还未能完成
    #     i, section_i = 0, 0   
    #     tmp_unit = copy.deepcopy(vec_unit)
    #     while i < 48:
    #         tri_vtx = []
    #         if i % 4 == 0:
    #             vec_unit = copy.deepcopy(tmp_unit)
    #             tmp_unit = np.matmul(self.rotate30_mat, vec_unit)
    #         section_i = i % 4 + 1

    #         vtx1 = pos + section_i * self.ob_unit * vec_unit
    #         vtx2 = pos + section_i * self.ob_unit * tmp_unit
    #         tri_vtx.append(pos)
    #         tri_vtx.append(vtx1)
    #         tri_vtx.append(vtx2)    # 三角区的三个顶点

    #         # 三角观察区有相交: 1)三角区与其他agent的相撞 2)三角区与道路边界的相撞
    #         if self._tri_rect_collision_test(tri_vtx, agent.get_vertice()) or self._tri_bound_collision_test(tri_vtx):
    #             state_feature[i] = pdata.O_MAX
    #             # 之后所有的三角形通行鼓励值都下降
    #             i += 1
    #             while (i % 4) != 0:
    #                 state_feature[i] = pdata.O_MAX 
    #                 i += 1
    #             i -= 1

    #         i += 1

    #     state_feature[48], state_feature[49] = vec[0], vec[1]
    #     state_feature[50], state_feature[51] = pos[0], pos[1]
    #     state_feature[52] = la.norm(pos - agent.get_destination())

    #     # if isinstance(agent, MotorVehicle):
    #     #     print('Hello agent MotorVehicle.')

    #     # elif isinstance(agent, NonMotorVehicle):
    #     #     print('...')

    #     # elif isinstance(agent, Pedestrian):
    #     #     print('...')

    #     # else:
    #     #     print("Parameter agent has to be one of class MotorVehicle, NonMotorVehicle and Pedestrian.")

    #     return state_feature


    def _get_state_feature(self, agent):
        rays = agent.get_rays()
        state = np.ones(pdata.STATE_DIMENSION)
        for i in range(0, len(rays)):
            crosspoints = geo.get_seg_ray_crosspoints(rays[i], edges)                       
            nearest = geo.get_nearest_distance(rays[i].s_point, crosspoints)
            if nearest > pdata.OBSERVATION_LIMIT:
                state[i] = pdata.OBSERVATION_LIMIT
            else:
                state[i] = nearest
            state[i] = state[i] / pdata.OBSERVATION_LIMIT   # Min-Max归一化

        start_idx = pdata.STATE_DIMENSION - 4
        agent_v, agent_des_local = agent.get_velocity(), agent.get_destination_local()
        state[start_idx+0] = (agent_v[0] + agent.get_max_velocity()) / (2* agent.get_max_velocity())     # Min-Max归一化
        state[start_idx+1] = (agent_v[1]+ agent.get_max_velocity()) / (2* agent.get_max_velocity())
        state[start_idx+2] = (agent_des_local[0] + 2*(pdata.LANE_W + pdata.LANE_L)) / (4*(pdata.LANE_L + pdata.LANE_W))
        state[start_idx+3] = (agent_des_local[1] + 2*(pdata.LANE_L+pdata.LANE_W)) / (4*(pdata.LANE_L + pdata.LANE_W))
        return state


    # when collision happened, return true
    # 需要逆时针排列的顶点坐标: tri_vtx.shape=(3,2), rect_vtx.shape=(4,2)
    # 先进行三边相交测试，再进行矩形顶点是否在三角形内测试
    def _tri_rect_collision_test(self, tri_vtx, rect_vtx):
        intersected = False
        # 1) 线段与矩形是否相交的测试 —— 若线段与多边形所有的边都没有交点，则不相交（可能在内部的情况不用考虑了，因为另两条边一定会相交）
        for i in range(0, len(tri_vtx)):
            seg = np.array([tri_vtx[i], tri_vtx[(i+1)%3]])
            if geo.segment_test(seg, rect_vtx):
                return True

        # 2) 顶点是否在三角形内，使用逆时针连接的三边做向量点积就可以了（因为没有可能出现三角形在矩形内部的情况）
        intersected = geo.vertex_test(tri_vtx, rect_vtx)
        return intersected

   
    # 检测观察区三角与马路边界的交叉
    def _tri_bound_collision_test(self, tri_vtx):
        # tri_vtx: a np.ndarray with shape of (3,2)
        for i in range(0, len(tri_vtx)):
            seg = np.array([tri_vtx[i], tri_vtx[(i+1)%3]])
            for j in range(0, 8):
                if geo.seg_seg_test(seg, edges[j]):
                    return True
        return False


    # shaped reward: 需要专家知识
    def _get_reward(self, agent, goal=None):
        reward = 0

        # 主线奖励：进入目的地、进入与目的地不符的区域、发生车辆行人碰撞，都会采用主线奖励
        if self._check_agent_collision(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent collided>\n')
            return reward
        # elif not self._check_bound(agent):  # 包括越出仿真区域、开到左车道都会直接终止
        #     reward = -5000
        #     self.logger.write_to_log('<agent out of bound>')
        #     return reward
        elif self._check_arrival(agent):
            reward = pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent arrived>\n')
            return reward
        elif self._check_outside_region(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent out>\n')
            return reward

        # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
        delta_d = la.norm(agent.get_last_position() - agent.get_destination_world()) - la.norm(agent.get_position() - agent.get_destination_world())
        _r_d = 10 * delta_d

        # 保持车道奖励
        if self._check_bound(agent):
            _r_lane_change = 10
        else:
            _r_lane_change = -10

        # 速度奖励
        velocity = agent.get_velocity()
        norm_v = la.norm(velocity)

        if isinstance(agent, motorVehicle):
            v_limit = pdata.VELOCITY_LIMIT
        elif isinstance(agent, Bicycle):
            v_limit = pdata.MAX_BICYCLE_VEL

        if  norm_v >= v_limit:
            _r_v = -10 * (norm_v - v_limit)
        elif norm_v:
            # cos奖励 —— 当前速度与目标点重合方向越贴近，奖励越高
            relative_pos = agent.get_destination_local()
            unit = np.array([0.0, 1.0])
            cosine = relative_pos.dot(unit) / (la.norm(relative_pos) + pdata.EPSILON)
            _r_v = 10 * cosine + 10 * norm_v  
        else:
            _r_v = 0

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
        # self.logger.write_to_log('reward: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity))
        # print('reward:' + str(reward))
        return reward 


    def _get_reward_pe(self, agent, goal=None):
        reward = 0

        # 主线奖励：进入目的地、进入与目的地不符的区域、发生车辆行人碰撞，都会采用主线奖励
        if self._check_agent_collision(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent collided>\n')
            return reward
        elif self._check_arrival(agent):
            reward = pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent arrived>\n')
            return reward
        elif self._check_outside_region(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent out>\n')
            return reward

        # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
        delta_d = la.norm(agent.get_last_position() - agent.get_destination_world()) - la.norm(agent.get_position() - agent.get_destination_world())
        _r_d = 10 * delta_d

        # 速度奖励
        velocity = agent.get_velocity()
        norm_v = la.norm(velocity)

        if  norm_v >= pdata.MAX_HUMAN_VEL:
            _r_v = -10 * (norm_v - pdata.MAX_HUMAN_VEL)
        elif norm_v:
            # cos奖励 —— 当前速度与目标点重合方向越贴近，奖励越高
            relative_pos = agent.get_destination_local()
            unit = np.array([0.0, 1.0])
            cosine = relative_pos.dot(unit) / (la.norm(relative_pos) + pdata.EPSILON)
            _r_v = 10 * cosine + 10 * norm_v  
        else:
            _r_v = 0

        # if isinstance(agent, MotorVehicle):     # 包导入路径的写法都会影响 isinstance()的准确性
        #     print('Hello agent MotorVehicle.')

        # elif isinstance(agent, NonMotorVehicle):
        #     print('...')

        # elif isinstance(agent, Pedestrian):
        #     print('...')

        # else:
        #     print("Parameter agent has to be one of class MotorVehicle, NonMotorVehicle and Pedestrian.")

        reward = _r_d + _r_v
        # print('reward: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity), file = pdata.EXPERIMENT_LOG)
        # self.logger.write_to_log('reward: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity))
        return reward 


    # 除了移动目标不一致，其余与 _get_reward 相同
    def get_her_reward(self, agent, normal_reward, future_pos):
        # future_pos 是相对坐标
        if abs(normal_reward) == pdata.MAIN_REWARD:
            reward = normal_reward
        else:
            # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
            delta_d = la.norm(future_pos)
            _r_d = 10 * delta_d

            # 保持车道奖励
            if self._check_bound(agent):
                _r_lane_change = 10
            else:
                _r_lane_change = -10

             # 速度奖励
            velocity = agent.get_velocity()
            norm_v = la.norm(velocity)
            if  norm_v >= pdata.VELOCITY_LIMIT:
                _r_v = -10 * (norm_v - pdata.VELOCITY_LIMIT)
            elif norm_v:
                # cos奖励 —— 与初始速度越接近，奖励越大
                origin_v = agent.get_origin_v()
                cosine = origin_v.dot(velocity) / (la.norm(origin_v) * norm_v) 
                _r_v = 10 * cosine + 10 * norm_v  
            else:
                _r_v = 0          

            reward  =_r_d + _r_lane_change + _r_v
            # self.logger.write_to_log('reward_her: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity))
        return reward


    # 检查是否终止
    def _check_termination(self, reward):
        if abs(reward) == pdata.MAIN_REWARD:
            return True
        else:
            return False
        

    # Agent间的碰撞检查 —— SAT算法
    # TODO: 针对多智能体碰撞的检测
    def _check_agent_collision(self, agent):
        collision = False
        _count = 0

        # 姑且采用暴力遍历方式检查相撞
        _count = len(self.vehicle_set)
        for i in range(0, _count):
            if agent == self.vehicle_set[i]:
                continue
            elif self._check_bilateral_collision(agent, self.vehicle_set):
                collision = True
        return collision


    # 矩形间的碰撞检测 —— 由于存在斜交的可能性，直接比较线段
    # TODO: 暴力遍历检查该怎么优化？
    def _check_bilateral_collision(self, agent1, agent_set):
        vertice1 = agent1.get_vertice()
        for i in range(0, len(agent_set)):
            vertice2 = agent_set[i].get_vertice()
            if geo.check_obb_collision(vertice1, vertice2):
                return True
        return False


    # 矩形是否行驶在合法区域内的碰撞检测 —— 顶点是否全部在合理行驶区域内
    def _check_bound(self, agent):
        # return False : if box is not inside the bound
        vertice = agent.get_vertice()
        _origin = agent.get_origin()
        _des = agent.get_des_string()
        if _origin == 'east' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['e_s'])
        elif _origin == 'east' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['e_w'])
        elif _origin == 'east' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['e_n'])  
        elif _origin == 'north' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['n_e'])
        elif _origin == 'north' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['n_s'])
        elif _origin == 'north' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['n_w'])
        elif _origin == 'west' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['w_s'])
        elif _origin == 'west' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['w_e'])
        elif _origin == 'west' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['w_n'])
        elif _origin == 'south' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['s_e'])
        elif _origin == 'south' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['s_n'])
        elif _origin == 'south' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['s_w'])


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

    # 是否到达目标点 —— 用距离来判断   2020-4-9:废弃
    # def _check_arrival(self, agent):
        # return True: if agent arrived their destination
        # distance = la.norm(agent.get_position() - agent.get_destination_world())  
        # if distance <= (agent.get_size_width()/2):
        #     return True
        # else:
        #     return False

    def _check_arrival(self, agent):
        # return True: if agent arrived their destination
        agent_cross = np.array([agent.get_position(),agent.get_last_position()])
        if isinstance(agent, vehicle):
            if agent.get_des_string() == 'south':
                return geo.seg_seg_test(agent_cross, des_seg['s'])
            elif agent.get_des_string() == 'east':
                return geo.seg_seg_test(agent_cross, des_seg['e'])
            elif agent.get_des_string() == 'north':
                return geo.seg_seg_test(agent_cross, des_seg['n'])
            elif agent.get_des_string() == 'west':
                return geo.seg_seg_test(agent_cross, des_seg['w'])

        if isinstance(agent, pedestrian):
            if agent.get_origin_edge() == 0:
                return geo.seg_seg_test(agent_cross, edges[7])
            elif agent.get_origin_edge() == 1:
                return geo.seg_seg_test(agent_cross, edges[2])
            elif agent.get_origin_edge() == 2:
                return geo.seg_seg_test(agent_cross, edges[1])
            elif agent.get_origin_edge() == 3:
                return geo.seg_seg_test(agent_cross, edges[4])
            elif agent.get_origin_edge() == 4:
                return geo.seg_seg_test(agent_cross, edges[4])
            elif agent.get_origin_edge() == 5:
                return geo.seg_seg_test(agent_cross, edges[6])
            elif agent.get_origin_edge() == 6:
                return geo.seg_seg_test(agent_cross, edges[5])
            elif agent.get_origin_edge() == 7:
                return geo.seg_seg_test(agent_cross, edges[0])


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
        for i in range(0, len(self.vehicle_set)):
            vertice = self.vehicle_set[i].get_vertice()
            info_list.append(vertice)   # 不能直接往numpy数组里添加元素，要先使用python原生list

        info_list = np.array(info_list)  
        return info_list      


    # 环境的 reset —— 重置单个agent
    def reset(self,agent):
        # print('\n -----------agent reset---------- \n')
        # self.logger.write_to_log('\n -----------agent reset---------- \n')
        agent.reset_agent()
        return self._get_state_feature(agent)


# 2020-5-22: 未用到
class GameEnvironment():
    vehicle_set = []
    pedestrian_set = []
    _game_vehicle_set = []
    _game_pedestrian_set = []

    # agent一旦加入环境，就不再更新参数，而是按之前训练好的策略逐帧刷新位置
    def add_agent_to_environment(self, agent):
        if isinstance(agent, vehicle):
            self.vehicle_set.append(agent)
        if isinstance(agent, pedestrian):
            self.pedestrian_set.append(agent)

    def generate_priority_queue(self):
        self.vehicle_set.sort(key = lambda x : x.enter_frame)
        self.pedestrian_set.sort(key = lambda x : x.enter_frame)

    def reset_environment(self):
        for i in range(0, len(self.vehicle_set)):
            self.vehicle_set[i].reset()
        for j in range(0, len(self.pedestrian_set)):
            self.pedestrian_set[j].reset()

    def reset(self,agent):
        # print('\n -----------agent reset---------- \n')
        # self.logger.write_to_log('\n -----------agent reset---------- \n')
        agent.reset_agent()
        return self._get_state_feature(agent)


    def step(self, agent, action):
        # 计算环境中的状态改变
        # for i in range(0, len(self._game_vehicle_set)):
        #     other_state = self._get_state_feature(self._game_vehicle_set[i], game_agent, self._game_vehicle_set, self._game_pedestrian_set)
        #     other_action = self._game_vehicle_set[i].select_action(other_state)
        #     self._game_vehicle_set[i].update_attr(other_action)
        # for i in range(0, len(self._game_pedestrian_set)):
        #     other_state = self._get_state_feature(self._game_pedestrian_set[i], game_agent, self._game_vehicle_set, self._game_pedestrian_set)
        #     other_action = self._game_pedestrian_set[i].select_action(other_state)
        #     self._game_pedestrian_set[i].update_attr(other_action)
        self.vehicle_set = copy.deepcopy(self._game_vehicle_set)
        self._game_vehicle_set.clear()
        self.pedestrian_set = copy.deepcopy(self._game_pedestrian_set)
        self._game_pedestrian_set.clear()
        agent.update_attr(action)
        next_state = self._get_state_feature(agent)
        if isinstance(agent, vehicle):
            reward = self._get_reward(agent)
        else:
            reward = self._get_reward_pe(agent)
        done = self._check_termination(reward)

        return next_state, reward, done


    # 仅仅是用于预计算，但不会真正改变环境的状态
    def game_step(self, agent, action):
        self._game_vehicle_set = copy.deepcopy(self.vehicle_set)
        self._game_pedestrian_set = copy.deepcopy(self.pedestrian_set)
        game_agent = copy.deepcopy(agent)

        # 计算环境中的状态改变
        for i in range(0, len(self._game_vehicle_set)):
            other_state = self._get_game_feature(self._game_vehicle_set[i], game_agent, self._game_vehicle_set, self._game_pedestrian_set)
            other_action = self._game_vehicle_set[i].select_action(other_state)
            self._game_vehicle_set[i].update_attr(other_action)
        for i in range(0, len(self._game_pedestrian_set)):
            other_state = self._get_game_feature(self._game_pedestrian_set[i], game_agent, self._game_vehicle_set, self._game_pedestrian_set)
            other_action = self._game_pedestrian_set[i].select_action(other_state)
            self._game_pedestrian_set[i].update_attr(other_action)

        game_agent.update_attr(action)
        # next_state = self._get_state_feature(game_agent,game_agent, self._game_vehicle_set, self._game_pedestrian_set)
        if isinstance(agent, vehicle):
            reward = self._get_reward(agent)
        else:
            reward = self._get_reward_pe(agent)
        # done = self._check_termination(reward)

        # return next_state, reward, done
        return reward


    # learner 对其他agent是不可见的
    def _get_game_feature(self, cur_agent, learner, game_vehicle_set, game_pedestrian_set):
        rays = cur_agent.get_rays()
        state = np.ones(pdata.STATE_DIMENSION)
        vtxs = []
        for j in range(0, len(game_vehicle_set)):
            if game_vehicle_set[j] == learner or game_vehicle_set[j] == cur_agent:
                continue
            vtxs.append(game_vehicle_set[j].get_vertice())
        for j in range(0, len(game_pedestrian_set)):
            if game_pedestrian_set[j] == learner or game_pedestrian_set[j] == cur_agent:
                continue
            vtxs.append(game_pedestrian_set[j].get_vertice())

        for i in range(0, len(rays)):
            crosspoints = geo.get_seg_ray_crosspoints(rays[i], edges)                       
            crosspoints_agent = geo.get_ray_box_crosspoints(rays[i], vtxs)
            crosspoints.extend(crosspoints_agent)
            nearest = geo.get_nearest_distance(rays[i].s_point, crosspoints)

            if nearest > pdata.OBSERVATION_LIMIT:
                state[i] = pdata.OBSERVATION_LIMIT
            else:
                state[i] = nearest
            state[i] = state[i] / pdata.OBSERVATION_LIMIT   # Min-Max归一化

        start_idx = pdata.STATE_DIMENSION - 4
        agent_v, agent_des_local = cur_agent.get_velocity(), cur_agent.get_destination_local()
        state[start_idx+0] = (agent_v[0] +  learner.get_max_velocity()) / (2* learner.get_max_velocity())     # Min-Max归一化
        state[start_idx+1] = (agent_v[1]+ learner.get_max_velocity()) / (2* learner.get_max_velocity())
        state[start_idx+2] = (agent_des_local[0] + 2*(pdata.LANE_W + pdata.LANE_L)) / (4*(pdata.LANE_L + pdata.LANE_W))
        state[start_idx+3] = (agent_des_local[1] + 2*(pdata.LANE_L+pdata.LANE_W)) / (4*(pdata.LANE_L + pdata.LANE_W))
        return state


    # TODO: 这里注释取消掉以后会有语法错误
    def _get_state_feature(self, agent):
        rays = agent.get_rays()
        state = np.ones(pdata.STATE_DIMENSION)
        vtxs = []
        # for j in range(0, len(self.vehicle_set)):
        #     if self.vehicle_set[j] == agent or self.vehicle_set[j] == agent:
        #         continue
        #     vtxs.append(self.vehicle_set[j].get_vertice())

        # for j in range(0, len(self.pedestrian_set)):
        #     if self.pedestrian_set[j] == agent or self.pedestrian_set[j] == agent:
        #         continue
        #     vtxs.append((self.pedestrian_set[j].get_vertice())

        for i in range(0, len(rays)):
            crosspoints = geo.get_seg_ray_crosspoints(rays[i], edges)                       
            crosspoints_agent = geo.get_ray_box_crosspoints(rays[i], vtxs)
            crosspoints.extend(crosspoints_agent)
            nearest = geo.get_nearest_distance(rays[i].s_point, crosspoints)
            if nearest > pdata.OBSERVATION_LIMIT:
                state[i] = pdata.OBSERVATION_LIMIT
            else:
                state[i] = nearest

        start_idx = pdata.STATE_DIMENSION - 4
        agent_v, agent_des_local = agent.get_velocity(), agent.get_destination_local()
        state[start_idx+0] = (agent_v[0] +  agent.get_max_velocity()) / (2* agent.get_max_velocity())     # Min-Max归一化
        state[start_idx+1] = (agent_v[1]+ agent.get_max_velocity()) / (2* agent.get_max_velocity())
        state[start_idx+2] = (agent_des_local[0] + 2*(pdata.LANE_W + pdata.LANE_L)) / (4*(pdata.LANE_L + pdata.LANE_W))
        state[start_idx+3] = (agent_des_local[1] + 2*(pdata.LANE_L+pdata.LANE_W)) / (4*(pdata.LANE_L + pdata.LANE_W))
        return state


    # shaped reward: 需要专家知识
    def _get_reward(self, agent, goal=None):
        reward = 0

        # 主线奖励：进入目的地、进入与目的地不符的区域、发生车辆行人碰撞，都会采用主线奖励
        if self._check_agent_collision(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent collided>\n')
            return reward
        # elif not self._check_bound(agent):  # 包括越出仿真区域、开到左车道都会直接终止
        #     reward = -5000
        #     self.logger.write_to_log('<agent out of bound>')
        #     return reward
        elif self._check_arrival(agent):
            reward = pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent arrived>\n')
            return reward
        elif self._check_outside_region(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent out>\n')
            return reward

        # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
        delta_d = la.norm(agent.get_last_position() - agent.get_destination_world()) - la.norm(agent.get_position() - agent.get_destination_world())
        _r_d = 10 * delta_d

        # 保持车道奖励
        if self._check_bound(agent):
            _r_lane_change = 10
        else:
            _r_lane_change = -10

        # 速度奖励
        velocity = agent.get_velocity()
        norm_v = la.norm(velocity)
        if  norm_v >= pdata.VELOCITY_LIMIT:
            _r_v = -10 * (norm_v - pdata.VELOCITY_LIMIT)
        elif norm_v:
            # cos奖励 —— 与初始速度越接近，奖励越大
            origin_v = agent.get_origin_v()
            cosine = origin_v.dot(velocity) / (la.norm(origin_v) * norm_v) 
            _r_v = 10 * cosine + 10 * norm_v  
        else:
            _r_v = 0

        reward = _r_d + _r_v + _r_lane_change
        return reward 


    def _get_reward_pe(self, agent, goal=None):
        reward = 0

        # 主线奖励：进入目的地、进入与目的地不符的区域、发生车辆行人碰撞，都会采用主线奖励
        if self._check_agent_collision(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent collided>\n')
            return reward
        elif self._check_arrival(agent):
            reward = pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent arrived>\n')
            return reward
        elif self._check_outside_region(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent out>\n')
            return reward

        # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
        delta_d = la.norm(agent.get_last_position() - agent.get_destination_world()) - la.norm(agent.get_position() - agent.get_destination_world())
        _r_d = 10 * delta_d

        # 速度奖励
        velocity = agent.get_velocity()
        norm_v = la.norm(velocity)
        if  norm_v >= pdata.MAX_HUMAN_VEL:
            _r_v = -10 * (norm_v - pdata.MAX_HUMAN_VEL)
        elif norm_v:
            # cos奖励 —— 与初始速度越接近，奖励越大
            origin_v = agent.get_origin_v()
            cosine = origin_v.dot(velocity) / (la.norm(origin_v) * norm_v) 
            _r_v = 10 * cosine + 10 * norm_v  
        else:
            _r_v = 0

        reward = _r_d + _r_v
        return reward 


    def _check_termination(self, reward):
        if abs(reward) == pdata.MAIN_REWARD:
            return True
        else:
            return False


    # Agent间的碰撞检查 —— SAT算法
    # TODO: 针对多智能体碰撞的检测
    def _check_agent_collision(self, agent):
        collision = False
        _count = 0

        # 姑且采用暴力遍历方式检查相撞
        _count = len(self.vehicle_set)
        for i in range(0, _count):
            if agent == self.vehicle_set[i]:
                continue
            elif self._check_bilateral_collision(agent, self.vehicle_set):
                collision = True
        return collision


    # 矩形间的碰撞检测 —— 由于存在斜交的可能性，直接比较线段
    # TODO: 暴力遍历检查该怎么优化？
    def _check_bilateral_collision(self, agent1, agent_set):
        vertice1 = agent1.get_vertice()
        for i in range(0, len(agent_set)):
            vertice2 = agent_set[i].get_vertice()
            if geo.check_obb_collision(vertice1, vertice2):
                return True
        return False
    
    
    # 矩形是否行驶在合法区域内的碰撞检测 —— 顶点是否全部在合理行驶区域内
    def _check_bound(self, agent):
        # return False : if box is not inside the bound
        vertice = agent.get_vertice()
        _origin = agent.get_origin()
        _des = agent.get_des_string()
        if _origin == 'east' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['e_s'])
        elif _origin == 'east' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['e_w'])
        elif _origin == 'east' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['e_n'])  
        elif _origin == 'north' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['n_e'])
        elif _origin == 'north' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['n_s'])
        elif _origin == 'north' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['n_w'])
        elif _origin == 'west' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['w_s'])
        elif _origin == 'west' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['w_e'])
        elif _origin == 'west' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['w_n'])
        elif _origin == 'south' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['s_e'])
        elif _origin == 'south' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['s_n'])
        elif _origin == 'south' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['s_w'])


    # 是否到达目标点 —— 用距离来判断 (2020-4-9: 废弃)
    # def _check_arrival(self, agent):
    #     # return True: if agent arrived their destination
    #     distance = la.norm(agent.get_position() - agent.get_destination_world())
    #     if distance <= (agent.get_size_width()/2):
    #         return True
    #     else:
    #         return False

    def _check_arrival(self, agent):
        # return True: if agent arrived their destination
        agent_cross = agent.get_position() - agent.get_last_position()
        if isinstance(agent, vehicle):
            if agent.get_des_string() == 'south':
                return geo.seg_seg_test(agent_cross, des_seg['s'])
            elif agent.get_des_string() == 'east':
                return geo.seg_seg_test(agent_cross, des_seg['e'])
            elif agent.get_des_string() == 'north':
                return geo.seg_seg_test(agent_cross, des_seg['n'])
            elif agent.get_des_string() == 'west':
                return geo.seg_seg_test(agent_cross, des_seg['w'])

        if isinstance(agent, pedestrian):
            if agent.get_origin_edge() == 0:
                return geo.seg_seg_test(agent_cross, edges[7])
            elif agent.get_origin_edge() == 1:
                return geo.seg_seg_test(agent_cross, edges[2])
            elif agent.get_origin_edge() == 2:
                return geo.seg_seg_test(agent_cross, edges[1])
            elif agent.get_origin_edge() == 3:
                return geo.seg_seg_test(agent_cross, edges[4])
            elif agent.get_origin_edge() == 4:
                return geo.seg_seg_test(agent_cross, edges[4])
            elif agent.get_origin_edge() == 5:
                return geo.seg_seg_test(agent_cross, edges[6])
            elif agent.get_origin_edge() == 6:
                return geo.seg_seg_test(agent_cross, edges[5])
            elif agent.get_origin_edge() == 7:
                return geo.seg_seg_test(agent_cross, edges[0])


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


class EnvironmentMA():
    def __init__(self):
        self.agent_queue = []

    def __del__(self):
        del self.agent_queue

    # 将agent的初始化放在环境之外进行
    def join_agent(self, agent_list):
        for ag in agent_list:
            self.agent_queue.append(ag)

    def reset(self):
        for ag in self.agent_queue:
            ag.reset()

    def step(self, action_list):
        self._update_environment(action_list)
        next_united_state = self.get_united_state_feature()
        rewards = self.get_united_reward()
        done = False    # TODO: 这里done的判定还需要写
        return next_united_state, rewards, done

    def update_policy(self, central_replay_buffer):
        for ag in self.agent_queue:
            ag.update_model(central_replay_buffer, self.agent_queue)

    def get_united_state_feature(self):
        united_state = []
        for ag in self.agent_queue:
            state = self._get_state_feature(ag)
            united_state.append(state)
        united_state = np.concatenate(united_state, axis=0)
        return united_state     # [s1, s2, ... sN]


    def _get_state_feature(self, agent):
        rays = agent.get_rays()
        state = np.ones(pdata.STATE_DIMENSION)
        for i in range(0, len(rays)):
            crosspoints = geo.get_seg_ray_crosspoints(rays[i], edges)                       
            nearest = geo.get_nearest_distance(rays[i].s_point, crosspoints)
            if nearest > pdata.OBSERVATION_LIMIT:
                state[i] = pdata.OBSERVATION_LIMIT
            else:
                state[i] = nearest
            state[i] = state[i] / pdata.OBSERVATION_LIMIT   # Min-Max归一化

        start_idx = pdata.STATE_DIMENSION - 4
        agent_v, agent_des_local = agent.get_velocity(), agent.get_destination_local()
        state[start_idx+0] = (agent_v[0] + agent.get_max_velocity()) / (2* agent.get_max_velocity())     # Min-Max归一化
        state[start_idx+1] = (agent_v[1]+ agent.get_max_velocity()) / (2* agent.get_max_velocity())
        state[start_idx+2] = (agent_des_local[0] + 2*(pdata.LANE_W + pdata.LANE_L)) / (4*(pdata.LANE_L + pdata.LANE_W))
        state[start_idx+3] = (agent_des_local[1] + 2*(pdata.LANE_L+pdata.LANE_W)) / (4*(pdata.LANE_L + pdata.LANE_W))
        return state


    def _update_environment(self, actions):
        for i in range(0, len(self.agent_queue)):
            agent = self.agent_queue[i]
            act = actions[i]
            agent.update_attr(act)


    # TODO：需要修改协作任务达成的判定
    def get_united_reward(self):
        united_reward = 0
        for ag in self.agent_queue:
            if isinstance(ag, vehicle):
                reward = self._get_reward(ag)
            elif isinstance(ag, pedestrian):
                reward = self._get_reward_pe(ag)
            united_reward = united_reward + reward
        return united_reward


    # shaped reward: 需要专家知识
    def _get_reward(self, agent, goal=None):
        reward = 0

        # 主线奖励：进入目的地、进入与目的地不符的区域、发生车辆行人碰撞，都会采用主线奖励
        if self._check_agent_collision(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent collided>\n')
            return reward
        # elif not self._check_bound(agent):  # 包括越出仿真区域、开到左车道都会直接终止
        #     reward = -5000
        #     self.logger.write_to_log('<agent out of bound>')
        #     return reward
        elif self._check_arrival(agent):
            reward = pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent arrived>\n')
            return reward
        elif self._check_outside_region(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent out>\n')
            return reward

        # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
        delta_d = la.norm(agent.get_last_position() - agent.get_destination_world()) - la.norm(agent.get_position() - agent.get_destination_world())
        _r_d = 10 * delta_d

        # 保持车道奖励
        if self._check_bound(agent):
            _r_lane_change = 10
        else:
            _r_lane_change = -10

        # 速度奖励
        velocity = agent.get_velocity()
        norm_v = la.norm(velocity)

        if isinstance(agent, motorVehicle):
            v_limit = pdata.VELOCITY_LIMIT
        elif isinstance(agent, Bicycle):
            v_limit = pdata.MAX_BICYCLE_VEL

        if  norm_v >= v_limit:
            _r_v = -10 * (norm_v - v_limit)
        elif norm_v:
            # cos奖励 —— 当前速度与目标点重合方向越贴近，奖励越高
            relative_pos = agent.get_destination_local()
            unit = np.array([0.0, 1.0])
            cosine = relative_pos.dot(unit) / (la.norm(relative_pos) + pdata.EPSILON)
            _r_v = 10 * cosine + 10 * norm_v  
        else:
            _r_v = 0

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
        # self.logger.write_to_log('reward: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity))
        # print('reward:' + str(reward))
        return reward 


    def _get_reward_pe(self, agent, goal=None):
        reward = 0

        # 主线奖励：进入目的地、进入与目的地不符的区域、发生车辆行人碰撞，都会采用主线奖励
        if self._check_agent_collision(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent collided>\n')
            return reward
        elif self._check_arrival(agent):
            reward = pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent arrived>\n')
            return reward
        elif self._check_outside_region(agent):
            reward = -pdata.MAIN_REWARD
            # self.logger.write_to_log('\n<agent out>\n')
            return reward

        # 接近奖励 —— 允许的最大速度的模也仅仅只有 0.69 m/frame
        delta_d = la.norm(agent.get_last_position() - agent.get_destination_world()) - la.norm(agent.get_position() - agent.get_destination_world())
        _r_d = 10 * delta_d

        # 速度奖励
        velocity = agent.get_velocity()
        norm_v = la.norm(velocity)
        
        if  norm_v >= pdata.MAX_HUMAN_VEL:
            _r_v = -10 * (norm_v - pdata.MAX_HUMAN_VEL)
        elif norm_v:
            # cos奖励 —— 当前速度与目标点重合方向越贴近，奖励越高
            relative_pos = agent.get_destination_local()
            unit = np.array([0.0, 1.0])
            cosine = relative_pos.dot(unit) / (la.norm(relative_pos) + pdata.EPSILON)
            _r_v = 10 * cosine + 10 * norm_v  
        else:
            _r_v = 0

        # if isinstance(agent, MotorVehicle):     # 包导入路径的写法都会影响 isinstance()的准确性
        #     print('Hello agent MotorVehicle.')

        # elif isinstance(agent, NonMotorVehicle):
        #     print('...')

        # elif isinstance(agent, Pedestrian):
        #     print('...')

        # else:
        #     print("Parameter agent has to be one of class MotorVehicle, NonMotorVehicle and Pedestrian.")

        reward = _r_d + _r_v
        # print('reward: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity), file = pdata.EXPERIMENT_LOG)
        # self.logger.write_to_log('reward: {r}  velocity: {v}m/frame'.format(r = reward, v = velocity))
        return reward 


    # 检查是否终止
    def _check_termination(self, reward):
        if abs(reward) == pdata.MAIN_REWARD:
            return True
        else:
            return False
        

    # Agent间的碰撞检查 —— SAT算法
    def _check_agent_collision(self, agent):
        collision = False
        _count = 0

        # 姑且采用暴力遍历方式检查相撞
        _count = len(self.agent_queue)
        for i in range(0, _count):
            if agent == self.agent_queue[i]:
                continue
            elif self._check_bilateral_collision(agent, self.agent_queue):
                collision = True
        return collision


    # 矩形间的碰撞检测 —— 由于存在斜交的可能性，直接比较线段
    # TODO: 暴力遍历检查该怎么优化？
    def _check_bilateral_collision(self, agent1, agent_set):
        vertice1 = agent1.get_vertice()
        for i in range(0, len(agent_set)):
            vertice2 = agent_set[i].get_vertice()
            if geo.check_obb_collision(vertice1, vertice2):
                return True
        return False


    # 矩形是否行驶在合法区域内的碰撞检测 —— 顶点是否全部在合理行驶区域内
    def _check_bound(self, agent):
        # return False : if box is not inside the bound
        vertice = agent.get_vertice()
        _origin = agent.get_origin()
        _des = agent.get_des_string()
        if _origin == 'east' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['e_s'])
        elif _origin == 'east' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['e_w'])
        elif _origin == 'east' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['e_n'])  
        elif _origin == 'north' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['n_e'])
        elif _origin == 'north' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['n_s'])
        elif _origin == 'north' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['n_w'])
        elif _origin == 'west' and _des == 'south':
            return geo.box_inside_polygon(vertice, polygons['w_s'])
        elif _origin == 'west' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['w_e'])
        elif _origin == 'west' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['w_n'])
        elif _origin == 'south' and _des == 'east':
            return geo.box_inside_polygon(vertice, polygons['s_e'])
        elif _origin == 'south' and _des == 'north':
            return geo.box_inside_polygon(vertice, polygons['s_n'])
        elif _origin == 'south' and _des == 'west':
            return geo.box_inside_polygon(vertice, polygons['s_w'])


    # 对 vehicle 来说，穿越路口就是成功；对行人来说，到达马路对面才是成功
    def _check_arrival(self, agent):
        # return True: if agent arrived their destination
        agent_cross = np.array([agent.get_position(),agent.get_last_position()])
        if isinstance(agent, vehicle):
            if agent.get_des_string() == 'south':
                return geo.seg_seg_test(agent_cross, intersection_des_seg['s'])
            elif agent.get_des_string() == 'east':
                return geo.seg_seg_test(agent_cross, intersection_des_seg['e'])
            elif agent.get_des_string() == 'north':
                return geo.seg_seg_test(agent_cross, intersection_des_seg['n'])
            elif agent.get_des_string() == 'west':
                return geo.seg_seg_test(agent_cross, intersection_des_seg['w'])
        if isinstance(agent, pedestrian):
            if agent.get_origin_edge() == 0:
                return geo.seg_seg_test(agent_cross, edges[7])
            elif agent.get_origin_edge() == 1:
                return geo.seg_seg_test(agent_cross, edges[2])
            elif agent.get_origin_edge() == 2:
                return geo.seg_seg_test(agent_cross, edges[1])
            elif agent.get_origin_edge() == 3:
                return geo.seg_seg_test(agent_cross, edges[4])
            elif agent.get_origin_edge() == 4:
                return geo.seg_seg_test(agent_cross, edges[4])
            elif agent.get_origin_edge() == 5:
                return geo.seg_seg_test(agent_cross, edges[6])
            elif agent.get_origin_edge() == 6:
                return geo.seg_seg_test(agent_cross, edges[5])
            elif agent.get_origin_edge() == 7:
                return geo.seg_seg_test(agent_cross, edges[0])
            

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
        for i in range(0, len(self.agent_queue)):
            vertice = self.agent_queue[i].get_vertice()
            info_list.append(vertice)   # 不能直接往numpy数组里添加元素，要先使用python原生list

        info_list = np.array(info_list)  
        return info_list    