import copy
import numpy as np 
from numpy import linalg as la
import public_data as pdata 
import DDPG as rl

# 三个相对转向的目的地坐标
relative_left = np.array([-pdata.LANE_L - pdata.LANE_W /2*3 , pdata.LANE_L + pdata.LANE_W/2])
relative_straight = np.array([pdata.LANE_W/2, 2*pdata.LANE_L + pdata.LANE_W*2])
relative_right = np.array([pdata.LANE_L + pdata.LANE_W/2, pdata.LANE_L + pdata.LANE_W/2])

# 四个出发点坐标
origin_east = np.array([pdata.LANE_L+pdata.LANE_W, pdata.LANE_W/2])
origin_west = np.array([-pdata.LANE_L-pdata.LANE_W, -pdata.LANE_W/2])
origin_north = np.array([pdata.LANE_W/2, pdata.LANE_L+pdata.LANE_W])
origin_south = np.array([pdata.LANE_W/2, -pdata.LANE_L-pdata.LANE_W])

# 四个初始出发位置对应的初始速度
origin_v_east = np.array([-pdata.VELOCITY_LIMIT / 10, 0.0])
origin_v_west = np.array([pdata.VELOCITY_LIMIT / 10, 0.0])
origin_v_north = np.array([0.0, -pdata.VELOCITY_LIMIT / 10])
origin_v_south = np.array([0.0, pdata.VELOCITY_LIMIT / 10])

# 四个从局部坐标系转全局坐标系的变换矩阵（2020.2.20）
east_transform_mat = np.array([
    [0.0, -1.0, pdata.LANE_L+pdata.LANE_W],         # 左上角的 2x2 矩阵，是局部坐标系相对世界坐标系的旋转矩阵，如这里相对世界系旋转了 90
    [1.0, 0.0, pdata.LANE_W/2],                     # [0,2] [1,2] 位置的数字分别对应局部系原点相对世界系原点的 x坐标 y坐标
    [0, 0, 1.0]])
north_transform_mat = np.array([
    [-1.0, 0.0, -pdata.LANE_W/2],
    [0.0, -1.0, pdata.LANE_W + pdata.LANE_L],
    [0, 0, 1.0]])
west_transform_mat = np.array([
    [0.0, 1.0, -pdata.LANE_L-pdata.LANE_W],
    [-1.0, 0.0, -pdata.LANE_W/2],
    [0, 0, 1.0]])
south_transform_mat = np.array([
    [1.0, 0.0, pdata.LANE_W/2],
    [0.0, 1.0, -pdata.LANE_L-pdata.LANE_W],
    [0.0, 0.0, 1.0]])

class Motor(object):
    # _property属性：类属性，可由该类的不同实例共享和改写，单下划线表明仅有类的实例及其子类实例可以访问
    _vector_shape = (2,)
    _epsilon = 0.000001
    _rotate90_mat = np.array([[0, -1],[1, 0]])

    def __init__(self):
        # 类对象的实例变量
        self._position = np.zeros((2))  # [x, y]
        self._last_position = self._position    # 保存上一次移动的位置
        self._velocity = np.zeros((2))  # [v_x, v_y]
        self._origin_v = np.zeros((2))  # 保存初始速度用于计算奖励
        self._destination = np.zeros((2))    # [x, y], 目的地就最好与特定方向解耦，尽量用相对方向
        # self.__acceleration = np.zeros((2))   # [φ, a], 相对当前速度矢量的转角，以及加速度的模
        # self.__policy_network = None  // 决定车辆运动的网络模型
        self._vertice_local = np.zeros((4,2))   # 记录四个顶点的局部坐标系，一旦初始化就不变更
        self._vertice_in_world = np.zeros((4, 2))   # 记录的是四个顶点在世界坐标系下的坐标
        self._origin = ''
        self._veer = '' 
        self._des_string = ''   # 用于记录最终目的地的字符串
        self.model = None       # 用于保存 DDPG 相关的类对象


    def __del__(self):
        del self._position
        del self._last_position
        del self._vector_shape
        del self._destination
        del self._epsilon
        del self._des_string
        tmp_str = self.__class__.__name__ +  ' is deleted'
        print(tmp_str, file = pdata.EXPERIMENT_LOG)

    
    # 设定初始出发点和速度
    def set_origin(self, origin):
        if not isinstance(origin, str):
            # print("set_origin function argument require an instance of class 'str' !")
            pdata.write_to_log("set_origin function argument require an instance of class 'str' !")
            return

        if origin == 'east':
            self._origin = 'east'
            self._position = origin_east
            self._velocity = origin_v_east
        elif origin == 'north':
            self._origin = 'north'
            self._position = origin_north
            self._velocity = origin_v_north
        elif origin == 'west':
            self._origin = 'west'
            self._position = origin_west
            self._velocity = origin_v_west
        elif origin == 'south':
            self._origin = 'south'
            self._position = origin_south
            self._velocity = origin_v_south

        self._origin_v = copy.deepcopy(self._velocity)
        log_str = 'agent initialtion: origin - {og}  position - {pos}  velocity - {v}m/frame'.format(v = self._velocity, og = self._origin, pos = self._position)
        # print(log_str)
        pdata.write_to_log(log_str)


    # 设置 agent 的转向 —— 这是为了让 agent 的运动与具体地形解耦
    def set_veer(self, des):
        if not isinstance(des, str):
            print("set_veer function argument require an instance of class 'str' !")
            return

        self._veer = des
        destination = np.zeros((3))
        # TODO: 这里的一大串是权宜之计
        if des == 'left':
            destination  = relative_left
            if self._origin == 'east':
                self._des_string = 'south'
            elif self._origin == 'north':
                self._des_string = 'east'
            elif self._origin == 'west':
                self._des_string = 'north'
            elif self._origin == 'south':
                self._des_string = 'west'
        elif des == 'straight':
            destination  = relative_straight
            if self._origin == 'east':
                self._des_string = 'west'
            elif self._origin == 'north':
                self._des_string = 'south'
            elif self._origin == 'west':
                self._des_string = 'east'
            elif self._origin == 'south':
                self._des_string = 'north'
        elif des == 'right':
            destination  = relative_right
            if self._origin == 'east':
                self._des_string = 'north'
            elif self._origin == 'north':
                self._des_string = 'west'
            elif self._origin == 'west':
                self._des_string = 'south'
            elif self._origin == 'south':
                self._des_string = 'east'

        # destination = np.ndarray.tolist(destination).append(1)  # 这样先 tolist() 再 append() 的链式操作会出现 destination 成为 None 的bug
        destination = np.ndarray.tolist(destination)
        destination.append(1)

        if self._origin == '':
            return
        elif self._origin == 'east':
            destination = np.matmul(east_transform_mat, destination)
        elif self._origin == 'north':
            destination = np.matmul(north_transform_mat, destination)
        elif self._origin == 'west':
            destination = np.matmul(west_transform_mat, destination)
        elif self._origin == 'south':
            destination = np.matmul(south_transform_mat, destination)

        self._destination = destination[0:2]    # 丢弃坐标运算添加的最后一个元素
        # print('destination position: {des}'.format(des = self._destination))
        pdata.write_to_log('destination position: {des}'.format(des = self._destination))


    # 对外部开放的 agent 初始化函数
    # TODO: 这里要模板化
    def initiate_agent(self, origin, veer):
        '''
        param origin: 一个表示出发方向的字符串，为 'west' 'north' 'east' 'south' 四选一
        param veer: 一个表示 agent 转向的字符串，为 'left' 'straight' 'right' 三选一
        '''
        self.set_origin(origin)
        self.set_veer(veer)
        self._vertice_in_world = self.calculate_vertice(self._position, self._velocity)
        # TODO: 这里的纯测试代码迟早要改的
        if origin == 'west' and veer =='straight': 
            self.model = rl.DDPG_Straight(pdata.STATE_DIMENSION, pdata.ACTION_DIMENSION, pdata.MAX_MOTOR_ACTION) 


    # 重置位置、速度、方向、顶点等原本就有的参数
    def reset_agent(self):
        self.set_origin(self._origin)
        self.set_veer( self._veer)
        self._vertice_in_world = self.calculate_vertice(self._position, self._velocity)

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

    def get_destination(self):
        des = copy.copy(self._destination)
        return des

    def get_vertice(self):
        vtx = copy.deepcopy(self._vertice_in_world)
        return vtx

    def get_des_string(self):
        des = copy.copy(self._des_string)
        return des

    def get_origin(self):
        og = copy.copy(self._origin)
        return og

    # def get_accelation(self):
    #     return self._acceleration

    def get_origin_v(self):
        v = copy.deepcopy(self._origin_v)
        return v

    def check_2darray(self, arr):
        if not isinstance(arr, np.ndarray):
            print('Function needs an numpy ndarray argument')
            return False
        elif arr.shape != self._vector_shape:
            print('The shape of input requires an 1D numpy array with 2 elements')
            return False
        else:
            return True

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

    # def set_acceleration(self, a):
    #     if not self.check_2darray(a):
    #         return
    #     else:
    #         self.__acceleration = a

    def set_destination(self, des):
        if not self.check_2darray(des):
            return
        else:
            self._destination = des


    # 剪切action的范围 —— 暂不用，在神经网络输出层已经有限制
    # def crap_acceleration(self, ac):
    #     # 这里根据学习，对于集合与类对象（可变对象），python总是自动处理为按引用传递
    #     if not self.check_2darray(ac):
    #         return
    #     elif (ac[0] - self._epsilon > 0) and (ac[0] - np.pi/2 > self._epsilon):
    #         ac[0] = np.pi / 2
    #     elif (ac[0] - self._epsilon < 0) and (np.pi/2 - ac[0] > self._epsilon):
    #         ac[0] = - np.pi / 2

    # 由加速度计算下一步长的速度
    def get_updated_velocity(self, ac):
        if not self.check_2darray(ac):
            return np.array([])
        else:
            v_t_ab = la.norm(self._velocity)
            v_sin = self._velocity[0] / v_t_ab
            v_cos = self._velocity[1] / v_t_ab

            ac_sin = np.sin(ac[0])
            ac_cos = np.cos(ac[0])
            # self.crap_acceleration(ac)
            v_delta = np.array([ac_cos*v_cos - ac_sin*v_sin , ac_sin*v_cos + ac_cos*v_sin]) * ac[1]
            v_next = v_delta + self._velocity

        ratio = pdata.MAX_VELOCITY / la.norm(v_next)    # 物理模型能达到的最大速度
        v_next = ratio * v_next
        return v_next


    # 由加速度更新 Agent 的属性
    def update_attr(self, action):
        v_next = self.get_updated_velocity(action)

        if len(v_next) == 0:
            print('action is invalid.')
            return
        else:
            self._velocity = v_next

        pos_next = self._position + v_next 
        self._last_position = copy.deepcopy(self._position)
        self._position = pos_next
        tmp_str = 'Agent Position: {pos}'.format(pos = self._position)
        print(tmp_str)
        pdata.write_to_log(tmp_str)
        self._vertice_in_world = self.calculate_vertice(pos_next, v_next)

    
    # 将局部坐标系内的顶点，由 pos（决定平移） 和 vec（决定朝向）计算出在世界坐标系下的坐标 
    def calculate_vertice(self, pos, vec):
        vertice = np.zeros((4, 2))
        if pos is None or vec is None:
            return vertice
        elif pos.shape != self._vector_shape or vec.shape != self._vector_shape:
            return vertice
        else:
            world_x = vec / la.norm(vec)    
            world_y = np.matmul(self._rotate90_mat, world_x)
            rotation_mat = np.array([[world_x[0], world_y[0]],[world_x[1], world_y[1]]])
            translation_vec = np.array([pos[0], pos[1]])
            for i in range(0, 4):
                tmp = np.matmul(rotation_mat, self._vertice_local[i])
                vertice[i] = tmp + translation_vec
            
        return vertice


    # 载入和保存模型参数的方式
    def load(self):
        pdata.write_to_log('.pth to be loaded...')
        self.model.load()


    def save(self):
        pdata.write_to_log('.pth to be saved...')
        self.model.save()


    # return numpy array
    def select_action(self, state):
        action = self.model.select_action(state)
        return action

    def add_to_replaybuffer(self, state, next_state, action, reward, done):
        self.model.replay_buffer.push((state, next_state, action, reward, done))

    def update_model(self):
        self.model.update()

    def get_buffer_storage(self):
        return self.model.replay_buffer.storage


class MotorVehicle(Motor):
    def __init__(self):
        super().__init__()
        self._vertice_local = np.array([[pdata.MOTER_L/2, -pdata.MOTOR_W/2],
        [pdata.MOTER_L/2, pdata.MOTOR_W/2],
        [-pdata.MOTER_L/2, pdata.MOTOR_W/2],
        [-pdata.MOTER_L/2, -pdata.MOTOR_W/2]])


class NonMotorVehicle(Motor):
    def __init__(self):
        super().__init__()
        self._vertice_local = np.array([[pdata.NON_MOTOR_L/2, -pdata.NON_MOTOR_W/2],
        [pdata.NON_MOTOR_L/2, pdata.NON_MOTOR_W/2],
        [-pdata.NON_MOTOR_LTER_L/2, pdata.NON_MOTOR_W/2],
        [-pdata.NON_MOTOR_L/2, -pdata.NON_MOTOR_W/2]])
        