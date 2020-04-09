from itertools import count
import os, sys, random, time  
# 在sys.path里动态添加了路径后，python才能找到项目中的模块
# 坑：在linux里使用 python 命令运行该.py 时，os.getcwd() 会随着用户当前所处的目录位置改变
sys.path.append(os.getcwd() + '/traffic_model/RL_models') 
import multiprocessing,threading
import json
import numpy as np
from numpy import linalg as la
import torch
from RL_models import environment as env
from RL_models import DDPG, log_util
from RL_models import public_data as pdata
from RL_models.vehicle import motorVehicle, Bicycle
from RL_models.pedestrian import pedestrian


class KGame(threading.Thread):
    # 10种有限预测
    actions_noise = np.array([[0.0, 1.0], [0.0, 0.5],[np.pi/6, 0.5],[np.pi/6, 1.0],[np.pi/3, 0.5],
     [np.pi/3, 1.0],[-np.pi/6, 0.5],[-np.pi/6, 1.0], [-np.pi/3, 0.5], [-np.pi/3, 1.0]])

    def __init__(self, k):
        self._k = k     # 博弈轮次数

    # 基于博弈的训练
    def training(self, learner): 
        # 收集训练数据
        if pdata.MODE == 'train':
            learner.load()     # 决定了是否导入已有的模型

            for i in range(pdata.MAX_EPISODE):
                self.roads.reset_environment()
                state = self.roads.reset(learner)      # state 的维数要尤其注意

                # count(): an infinite iterator provided by module itertools
                # 用于往经验池中放入样本的内层循环
                for t in count():
                    action = learner.select_action(state)
                    max_reward = float('-inf')
                    tmp_action = np.zeros((2,))

                    # 选择最大收益的动作调整执行，并放入样本池
                    for a_idx in range(0, len(self.actions_noise)):
                        noise = self.actions_noise[a_idx]
                        game_action = np.array([action[0] + noise[0], action[1] * noise[1]])

                        if(abs(game_action[0]) > pdata.MAX_MOTOR_ACTION[0]):
                            game_action[0] = pdata.MAX_MOTOR_ACTION[0] if game_action[0] > pdata.MAX_MOTOR_ACTION[0] else -pdata.MAX_MOTOR_ACTION[0]
                        if(abs(game_action[1]) > pdata.MAX_MOTOR_ACTION[1]):
                            game_action[1] = pdata.MAX_MOTOR_ACTION[1] if game_action[1] > pdata.MAX_MOTOR_ACTION[1] else -pdata.MAX_MOTOR_ACTION[1]
                        
                        reward = self.roads.game_step(learner, game_action)     # 与step一样返回，但不会实际改变环境状态
                        if reward > max_reward:
                            tmp_action = game_action
                    # if pdata.RENDER and i >= pdata.RENDER_INTERVAL: roads.render()
                    action =  tmp_action
                    next_state, reward, done = self.roads.step(learner, action)
                    # 往经验池里增加序列元素
                    # learner.add_to_replaybuffer(tmp_seq[0][0], tmp_seq[0][1], tmp_seq[0][2], tmp_seq[0][3], tmp_seq[0][4]) 
                    learner.add_to_replaybuffer(state, next_state, action, reward, done)
                    state = next_state
                    if done or t >= pdata.MAX_LENGTH_OF_TRAJECTORY:
                        break

                # self.logger.add_to_position_seq()

                if i % pdata.LOG_INTERVAL== 0:
                    # print('motor_thread saves parameters.')
                    learner.save_as(1)   # 这是 level-k 级的博弈

                # 样本池满了之后每轮迭代都在更新模型
                if len(learner.get_buffer_storage()) >= pdata.CAPACITY - 1:
                    # self.logger.write_to_log('params update!\n')
                    learner.update_model()  # 更新


    def run(self):
        self.logger_env = log_util.logWriter('env')
        self.logger = log_util.logWriter('k-game-1')
        car_east_left = motorVehicle(self.logger_env)    # TODO:这里要载入已有的左转模型 —— 文件里先载入
        mark_name = ''                                   # file_actor = pdata.DIRECTORY +  mark_str + '_actor.pth'
        car_east_left.load_from(mark_name)
        car_learner = motorVehicle( self.logger) 
        self.roads = env.GameEnvironment()
        self.roads.add_agent_to_environment(car_east_left)  # car_east_left 当背景板去了
        self.training(car_learner)


if __name__ == '__main__':
    thread_car = KGame(1)
    thread_car.start()
    thread_car.join()
    print('main process ends.')
