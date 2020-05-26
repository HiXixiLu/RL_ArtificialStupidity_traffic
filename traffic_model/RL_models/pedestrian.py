import copy
import numpy as np 
from numpy import linalg as la
import public_data as pdata 
import geometry as geo
import DDPG as rl


rotate_nag90_mat = np.array([[0.0, 1.0], [-1.0, 0.0]])

# 行人仅有一种行为就是横穿马路
class pedestrian():
    _vector_shape = _vector_shape = (2,)
    _rotate90_mat = np.array([[0, -1],[1, 0]])

    def __init__(self, logger):
        # 类对象的实例变量
        self._position = np.zeros((2))  # [x, y]
        self._last_position = self._position    # 保存上一次移动的位置
        self._origin_v = np.zeros((2))  # 保存初始速度用于计算奖励
        self._destination_world = np.zeros((2))    
        self._destination_local = np.array([0.0, 2*pdata.LANE_W - 2*pdata.P_W])
        self._vertice_local = np.array([[pdata.P_L/2, -pdata.P_W/2],
        [pdata.P_L/2, pdata.P_W/2],
        [-pdata.P_L/2, pdata.P_W/2],
        [-pdata.P_L/2, -pdata.P_W/2]])   # 记录四个顶点的局部坐标系，一旦初始化就不变更
        self._vertice_in_world = np.zeros((4, 2))   # 记录的是四个顶点在世界坐标系下的坐标
        self._origin_pos = np.zeros((2,2))       # 记录行人的出发点
        self._velocity = np.ones((2,2))
        self.model = None       # 用于保存 DDPG 相关的类对象
        self._width = pdata.P_W
        self._length = pdata.P_L
        self.logger = logger
        self._edge = 0      # 用于记录出发边界
        self._distance = 0.0    # 用于记录出发点距离路口的直线距离
        self._rays = []          # 依据 a*x + b*y = c 的直线方程保存三个系数元组 a,b,c
        self.enter_frame = 0


    def check_2darray(self, arr):
        if not isinstance(arr, np.ndarray):
            print('Function needs an numpy ndarray argument')
            return False
        elif arr.shape != self._vector_shape:
            print('The shape of input requires an 1D numpy array with 2 elements')
            return False
        else:
            return True

    
    # 将局部坐标系内的顶点，由 pos（决定平移） 和 vec（决定朝向）计算出在世界坐标系下的坐标 
    def calculate_vertice(self, pos, vec):
        vertice = np.zeros((4, 2))
        if pos is None or vec is None:
            return vertice
        elif pos.shape != self._vector_shape or vec.shape != self._vector_shape:
            return vertice
        else:
            world_x = vec / (la.norm(vec) + pdata.EPSILON)    
            world_y = np.matmul(self._rotate90_mat, world_x)
            rotation_mat = np.array([[world_x[0], world_y[0]],[world_x[1], world_y[1]]])
            translation_vec = np.array([pos[0], pos[1]])
            for i in range(0, 4):
                tmp = np.matmul(rotation_mat, self._vertice_local[i])
                vertice[i] = tmp + translation_vec       
        return vertice

    
    # 又来了粗暴式实现
    # down-left-horizontal, down-left-vertical et al
    def set_origin(self, edge, distance):
        # edge: 8个边界的index
        # distance: 距离路口的距离 <= 18m
        if not isinstance(edge, int) or not isinstance(distance, float):
            print('set_origin: invalid arguments')
            return

        if edge == 0:
            self._origin_pos = np.array([-pdata.LANE_W-distance, -pdata.LANE_W + self._width/2])
            self._destination_world = self._origin_pos + np.array([0.0, 2 * pdata.LANE_W])
            self._origin_v = np.array([0.0, 1.0 / pdata.FPS])
        elif edge == 1:
            self._origin_pos = np.array([-pdata.LANE_W + self._width/2, -pdata.LANE_W-distance])
            self._destination_world = self._origin_pos + np.array([2 * pdata.LANE_W, 0.0])
            self._origin_v = np.array([1.0 / pdata.FPS, 0.0])
        elif edge == 2:
            self._origin_pos = np.array([pdata.LANE_W - self._width/2, -pdata.LANE_W-distance])
            self._destination_world = self._origin_pos + np.array([-2 * pdata.LANE_W, 0.0])
            self._origin_v = np.array([-1.0/pdata.LANE_W , 0.0])
        elif edge == 3:
            self._origin_pos = np.array([pdata.LANE_W + distance, -pdata.LANE_W + self._width/2])
            self._destination_world = self._origin_pos + np.array([0.0, 2* pdata.LANE_W])
            self._origin_v = np.array([0.0, 1.0/pdata.FPS])
        elif edge == 4:
            self._origin_pos = np.array([pdata.LANE_W + distance, pdata.LANE_W - self._width/2])
            self._destination_world = self._origin_pos + np.array([0.0, -2*pdata.LANE_W])
            self._origin_v = np.array([0.0, -1.0/pdata.FPS])
        elif edge == 5:
            self._origin_pos = np.array([pdata.LANE_W - self._width /2, pdata.LANE_W + distance])
            self._destination_world = self._origin_pos + np.array([-2*pdata.LANE_W, 0.0])
            self._origin_v = np.array([-1.0/pdata.LANE_W, 0.0])
        elif edge == 6:
            self._origin_pos = np.array([-pdata.LANE_W+self._width/2, pdata.LANE_W + distance])
            self._destination_world = self._origin_pos + np.array([2*pdata.LANE_W, 0.0])
            self._origin_v = np.array([1.0/pdata.LANE_W, 0.0])
        elif edge == 7:
            self._origin_pos = np.array([-pdata.LANE_W-distance, pdata.LANE_W - self._width/2])
            self._destination_world = self._origin_pos + np.array([0.0, -2*pdata.LANE_W])
            self._origin_v = np.array([0.0, -1.0/pdata.LANE_W])  

        self._edge = edge
        self._distance = distance
        self._velocity = copy.deepcopy(self._origin_v)
        self._position = copy.deepcopy(self._origin_pos)


    def set_rays(self):
        unit_v = self._velocity / (la.norm(self._velocity) + pdata.EPSILON)
        origin_ray = geo.ray(self._position, unit_v[0], unit_v[1])
        self._rays.append(copy.deepcopy(origin_ray))
        for i in range(1, 12):
            origin_ray = geo.get_rotation_thirty_ray(origin_ray)
            self._rays.append(copy.deepcopy(origin_ray))


    def _update_rays(self):
        unit_v = self._velocity / (la.norm(self._velocity) + pdata.EPSILON)  #[cosine, sine]
        tmp_ray = geo.ray(self._position, unit_v[0], unit_v[1])
        for i in range(0,12):
            self._rays[i] = copy.deepcopy(tmp_ray)
            tmp_ray = geo.get_rotation_thirty_ray(tmp_ray)   


    def initiate_agent(self, origin_edge, distance, isHER = False, enter_frame=0):
        self.set_origin(origin_edge, distance)
        self.set_rays()   
        self._vertice_in_world = self.calculate_vertice(self._position, self._origin_v)  
        self.enter_frame = enter_frame      
        if isHER == True:
            self.model = rl.DDPG_PE_HER(pdata.STATE_HER_DIMENSION, pdata.ACTION_DIMENSION, pdata.MAX_HUMAN_VEL, self.logger)
        else:
            self.model = rl.DDPG_PE(pdata.STATE_DIMENSION, pdata.ACTION_DIMENSION, pdata.MAX_HUMAN_VEL, self.logger)


    # 重置位置、速度、方向、顶点等原本就有的参数
    def reset_agent(self):
        self.set_origin(self._edge, self._distance)
        self._vertice_in_world = self.calculate_vertice(self._position, self._origin_v)
        self._update_destination_local()
        self._update_rays()


    # 根据python的传参特性，这里必须保证返回的是一个拷贝值
    def get_position(self):
        pos = copy.deepcopy(self._position)
        return pos

    def get_last_position(self):
        last_pos = copy.deepcopy(self._last_position)
        return last_pos

    def get_velocity(self):
        ve = copy.deepcopy(self._velocity)
        return ve

    def get_destination_world(self):
        des = copy.deepcopy(self._destination_world)
        return des

    def get_destination_local(self):
        des = copy.deepcopy(self._destination_local)
        return des

    def get_vertice(self):
        vtx = copy.deepcopy(self._vertice_in_world)
        return vtx

    def get_origin(self):
        og = copy.copy(self._origin_pos)
        return og

    def get_max_velocity(self):
        return pdata.MAX_HUMAN_VEL

    # def get_accelation(self):
    #     return self._acceleration

    def get_origin_v(self):
        v = copy.deepcopy(self._origin_v)
        return v


    def get_size_width(self):
        wid = copy.deepcopy(self._width)
        return wid

    def get_size_length(self):
        leng = copy.deepcopy(self._length)
        return leng

    def get_rays(self):
        rays = copy.deepcopy(self._rays)
        return rays

    def get_origin_edge(self):
        return self._edge

    def _set_position(self, pos):
        if not self.check_2darray(pos):
            return
        else:
            self._position = copy.deepcopy(pos)

    def _set_velocity(self, v):
        if not self.check_2darray(v):
            return
        else:
            self._velocity = copy.deepcopy(v)

    def get_modify_ratio(self, vel):
        return pdata.MAX_HUMAN_VEL / (la.norm(vel) + pdata.EPSILON)

    # 特殊地，允许行人往回跑
    def update_attr(self, vel):
        v_next = vel + self._velocity
        ratio = self.get_modify_ratio(v_next)
        if ratio < 1:
            v_next = ratio * v_next

        self._set_velocity(v_next)
        pos_next = self._position + v_next 
        self._last_position = copy.deepcopy(self._position)
        self._set_position(pos_next)
        self._update_destination_local()
        self._update_rays()
        # tmp_str = 'Agent Position: {pos}'.format(pos = self._position)
        # self.logger.write_to_log(tmp_str)
        self.logger.record_position(self._position)

    # 更新相对坐标
    def _update_destination_local(self):  
        relative_des_local = geo.world_to_local(self._position, self._velocity, self._destination_world)
        self._destination_local = relative_des_local


    # 载入和保存模型参数的方式
    def load(self):
        # self.logger.write_to_log('.pth to be loaded...')
        self.model.load(pdata.AGENT_TUPLE[2])


    def save(self):
        # self.logger.write_to_log('.pth to be saved...')
        self.model.save(pdata.AGENT_TUPLE[2])


    # return numpy array
    def select_action(self, state):
        action = self.model.select_action(state)
        return action

    def normalize_action(self, action):
        if isinstance(action, np.ndarray) and self.check_2darray(action):
            action = (action +  pdata.MAX_HUMAN_VEL) / (2 * pdata.MAX_HUMAN_VEL)

    def add_to_replaybuffer(self, state, next_state, action, reward, done):
        self.normalize_action(action)
        self.model.replay_buffer.push((state, next_state, action, reward, done))

    def add_to_filter_repleybuffer(self, data_seq):
        # data_seq : [[next_state, state, action, reward, done]...]
        for i in range(0, len(data_seq)):
            action = data_seq[i][2]
            self.normalize_action(action)
        self.model.replay_buffer.push(data_seq)

    def update_model(self):
        self.model.update()

    def get_buffer_storage(self):
        return self.model.replay_buffer.storage


class PedestrianMA(pedestrian):
    def __init__(self, logger): 
        super().__init__(logger)
        self.arrived = False

    def update_model(self, central_replay_buffer, agent_list):
        if self.arrived:
            return
        for idx in range(len(agent_list)):
            if agent_list[idx] == self:
                self.model.update(central_replay_buffer, agent_list, idx)
                break

    def select_target_actions(self, states):
        # state 的输入必须是归一化的[[],[]] np.ndarray
        action = self.model.select_target_actions(states)
        for act in action:
            self.normalize_action(act)
        return action   # 归一化的

    def select_current_actions(self, states):
        action = self.model.select_current_actions(states)
        for act in action:
            self.normalize_action(act)
        return action   # 归一化的

    def initiate(self, origin_edge, distance, agent_n):
        self.set_origin(origin_edge, distance)
        self._origin_edge = origin_edge
        self._distance_to_intersection = distance
        self.set_rays()
        self._vertice_in_world = self.calculate_vertice(self._position, self._origin_v)
        self.enter_frame = 0
        self.model = rl.MADDPG_PE(pdata.STATE_DIMENSION, pdata.ACTION_DIMENSION, pdata.MAX_HUMAN_VEL, agent_n, self.logger)

    def reset(self):
        self.set_origin(self._origin_edge, self._distance_to_intersection)
        self._vertice_in_world = self.calculate_vertice(self._position, self._origin_v)
        self._update_destination_local()
        self._update_rays()

    def save(self):
        # self.logger.write_to_log('Bicycle: .pth to be saved...')
        self.model.save(pdata.AGENT_TUPLE[2]+'_MA')

    def load(self):
        # self.logger.write_to_log('Bicycle: .pth to be loaded...')
        self.model.load(pdata.AGENT_TUPLE[2]+'_MA') 