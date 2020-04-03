import numpy as np
import os
import time

''' 该模块是一个纯粹的数据块，这算不算单例模式？ '''
FPS = 24                                    # 刷新的帧率，即每刷新一次速度的改变
VELOCITY_LIMIT = 16.667 / FPS                    # 由 60km/h 换算成 m/frame
MAX_VELOCITY = 33.333 / FPS                      # 由 120km/h 的汽车最高时速换算成 m/frame
MAX_ACCELERATION = 2.315 / FPS               # 机动车的每帧加速度(百公里加速12s折算而来)
MAX_HUMAN_AC = 2.0 / FPS                        # 一般单速自行车的最大加速度为 2m/s^2
MAX_HUMAN_VEL = 10.0 / FPS                       # 人类的极限速度是 16m/s，这里取10
LANE_L, LANE_W,  = 18.0, 3.0                # 道路数据都改成浮点数，避免隐式类型转换的坑。单位: m，m，m/s
MOTER_L, MOTOR_W = 3.8, 1.8                 # 机动车尺寸单位: m，m
NON_MOTOR_L, NON_MOTOR_W = 1.6, 0.5         # 自行车尺寸单位: m，m （对应26寸自行车，及较为宽裕的骑手宽度）
P_L, P_W = 0.4, 0.4                         # 行人尺寸单位: m，m
O_MAX = -10.0                             # 设定的被占用的伪数据 —— 通行鼓励值
 # 在tanh的归一化函数中，上下限是[-1, 1]；1.6L 排量的百公里加速一般有12s —— 换算每秒均加速 2.3148148148m/s，再换算为每帧加速度
MAX_MOTOR_ACTION = np.array([np.pi/2, MAX_ACCELERATION])    # 2020-3-26：从物理模型本身防止开倒车
MAX_BICYCLE_ACTION = np.array([np.pi/2, MAX_HUMAN_AC])
VEER_TUPLE = ('straight', 'left', 'right')
DIRECTION_TUPLE = ('west', 'south', 'east', 'north')
AGENT_TUPLE = ('motor', 'bicycle', 'pedestrian')


''' 以下是供训练使用的参数 '''
MODE = 'train'
OBSERVATION_DIMENSION = 12   # 距离观察维度
OBSERVATION_LIMIT = 2 * (LANE_L + LANE_W)      # 最大观察距离：m
# STATE_DIMENSION = 53    # 见自己的设计文档
STATE_DIMENSION = OBSERVATION_DIMENSION + 4    # 见毕业论文
STATE_HER_DIMENSION = STATE_DIMENSION + 2   # 2: 作为 goal 的position维度
ACTION_DIMENSION = 2
MAIN_REWARD = 300
EPSILON = 1e-6

TAU = 0.005     # target smoothing coefficient：从current网络往target网络里更新参数用
TARGET_UPDATE_INTERVAL = 1
TEST_ITERATION = 10
LEARNING_RATE = 1e-3     # 0.001的学习率
GAMMA = 0.99    # discounted factor
CAPACITY = 50000   # replay buffer size
BATCH_SIZE = 64  # minimun batch size
SEED = False
RANDOM_SEED = 9527
''' optional parameters '''
SAMPLE_FREQUENCY = 256
RENDER = False
LOG_INTERVAL = 50
LOAD = True    # load model
RENDER_INTERVAL = 100
EXPLORATION_NOISE = 0.3     # 探索率
ANGLE_SD = (EXPLORATION_NOISE * np.pi/2) / 2.576     # 正态分布的标准差σ：满足 X~N(0, σ^2) 使上α分位点在 0.2 * np.pi/2 上，其中α=0.005
NORM_SD = MAX_ACCELERATION / 2.576  # 正态分布的标准差σ：满足 X~N(0, σ^2) 使上α分位点在 0.2 * MAX_ACCELERATION，其中α=0.005
MAX_EPISODE = 58000    # num of games —— 进行实验的次数，也即Montecarlo序列的采集数
MAX_LENGTH_OF_TRAJECTORY = 500     # num of frames —— 单次训练最大序列长度
PRINT_LOG = 5
UPDATE_ITERATION = 10   # 一次网络参数更新的均值计算次数
HER_K = 8   # 1/8 of BATCH_SIZE

''' 文件的保存路径 '''
DIRECTORY = os.getcwd() +'/traffic_model/data/'  # python文件路径可处理适应于不同系统 