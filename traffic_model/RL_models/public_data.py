import numpy as np
import time

''' 该模块是一个纯粹的数据块，这算不算单例模式？ '''
FPS = 24                                    # 刷新的帧率，即每刷新一次速度的改变
VELOCITY_LIMIT = 16.667 / FPS                    # 由 60km/h 换算成 m/frame
MAX_ACCELERATION = 27.778/FPS
LANE_L, LANE_W,  = 18.0, 3.0                # 道路数据都改成浮点数，避免隐式类型转换的坑。单位: m ，m，m/s
MOTER_L, MOTOR_W = 3.8, 1.8                 # 机动车尺寸单位: m，m
NON_MOTOR_L, NON_MOTOR_W = 1.5, 0.6         # 非机动车尺寸单位: m，m
P_L, P_W = 0.4, 0.4                         # 行人尺寸单位: m，m
O_MAX = -10.0                             # 设定的被占用的伪数据 —— 通行鼓励值
MAX_MOTOR_ACTION = np.array([np.pi, MAX_ACCELERATION])          # 在tanh的归一化函数中，上下限是[-1, 1]；1.6L 排量的百公里加速一般有12s —— 换算每秒均加速 27.77777m/s，再换算为每帧加速度


''' 以下是供训练使用的参数 '''
MODE = 'train'
STATE_DIMENSION = 53    # 见自己的设计文档
ACTION_DIMENSION = 2
TAU = 0.005     # target smoothing coefficient：从current网络往target网络里更新参数用
TARGET_UPDATE_INTERVAL = 1
TEST_ITERATION = 10
LEARNING_RATE = 1e-3     # 0.001的学习率
GAMMA = 0.99    # discounted factor
CAPACITY = 2000    # replay buffer size
BATCH_SIZE = 64  # minimun batch size
SEED = False
RANDOM_SEED = 9527
''' optional parameters '''
SAMPLE_FREQUENCY = 256
RENDER = False
LOG_INTERVAL = 50
LOAD = True    # load model
RENDER_INTERVAL = 100
EXPLORATION_NOISE = 1.0  # 随机探索过程的噪声率/正态分布的标准差σ
MAX_EPISODE = 20000    # num of games —— 进行实验的次数，也即Montecarlo序列的采集数
MAX_LENGTH_OF_TRAJECTORY = 1000     # num of frames —— 单次训练最大序列长度
PRINT_LOG = 5
UPDATE_ITERATION = 10   # 一次网络参数更新的均值计算次数
''' 文件的保存路径 '''
DIRECTORY = 'D:\\python_project\\TrafficModel\\traffic_model\\data\\'  # 该文件路径只能这么暂时写死，这是典型的windows路径写法

''' 文件生成的时间 '''
START_TIME = time.strftime("%Y-%m-%d_%H-%M",time.localtime(time.time()))
EXPERIMENT_LOG = open(DIRECTORY + START_TIME + '.txt', 'a+')

def write_to_log(content_str):
    print(content_str, file = EXPERIMENT_LOG)

def close_file():
    EXPERIMENT_LOG.close()

def record_hard_params():
    write_to_log('test')
    write_to_log('===============  initialization parameters  ===============')
    write_to_log('experiment: episodes = {e}, len_of_trajectory = {lt}, gamma = {g}, learning_rate = {lr}, exploration_rate = {er},\n buffer_size = {bs}, batch_size = {bs1}'.format(
        e = MAX_EPISODE, lt = MAX_LENGTH_OF_TRAJECTORY, g = GAMMA, lr = LEARNING_RATE, er = EXPLORATION_NOISE, bs = CAPACITY, bs1 = BATCH_SIZE
    ))
    write_to_log('environment limits : max_velocity = {mv} m/frame, fps = {fps} frames/s'.format(
        mv = VELOCITY_LIMIT, fps = FPS
    ))
    write_to_log('===========================================================')

# def record_agent_params(len, wid, og, des):
#     write_to_log('====================  agent parameters  ===================')
#     write_to_log('agent params: L = {l}m, W = {w}m, origin = {o}, destination = {d}'.format(
#         l = len, w = wid, o = og, d = des
#     ))
#     write_to_log('===========================================================')
    