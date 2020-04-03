from itertools import count
import os, sys, random, time  
# 在sys.path里动态添加了路径后，python才能找到项目中的模块
# 坑：在linux里使用 python 命令运行该.py 时，os.getcwd() 会随着用户当前所处的目录位置改变
sys.path.append(os.getcwd() + '/traffic_model/RL_models') 
import multiprocessing,threading
import json
import numpy as np
import torch
from RL_models import environment as env
from RL_models import DDPG, log_util
from RL_models import public_data as pdata
from RL_models.vehicle import motorVehicle, Bicycle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
min_Val = torch.tensor(1e-7).float().to(device) # min value : .to(device) 表示由 GPU 训练


# 这里实现的是HER的 future 模式
# class motor_train_process_HER(multiprocessing.Process):
class motor_train_process_HER(threading.Thread):
    def __init__(self, origin_idx, veer_idx, agent_idx):
        super().__init__()
        # self.daemon = True
        self._origin_idx = origin_idx
        self._veer_idx = veer_idx
        mark_str = pdata.AGENT_TUPLE[agent_idx] + '_' + pdata.DIRECTION_TUPLE[origin_idx]+'_'+pdata.VEER_TUPLE[veer_idx]
        self.logger = log_util.logWriter(mark_str+'_HER')
        self.roads = env.TrainingEnvironment(self.logger)

        if agent_idx == 0:
            self.agent = motorVehicle(self.logger)
        elif agent_idx == 1:
            self.agent = Bicycle(self.logger)
        # elif vehicle_str == pdata.AGENT_TUPLE[2]:
        #     self.agent == pedestrian()

    # 单agent的线程
    def run(self):
        # origin_idx: 小车的出发地   veer_idx 小车的转向
        self.agent.initiate_agent(pdata.DIRECTION_TUPLE[self._origin_idx], pdata.VEER_TUPLE[self._veer_idx], isHER = True)    

        # 收集训练数据
        if pdata.MODE == 'train':
            # self.logger.record_start_time()
            if pdata.LOAD: 
                self.agent.load()     # 决定了是否导入已有的模型

            final_goal = self.agent.get_destination_local() #相对坐标

            for i in range(pdata.MAX_EPISODE):
                state = self.roads.reset(self.agent)      # state 的维数要尤其注意               
                future_pos = np.zeros(2)

                for t in count():
                    state_her = np.concatenate([state, final_goal], axis = 0)
                    action = self.agent.select_action(state_her)

                    # issue 3 add noise to action 
                    noise = np.array([np.random.normal(0, pdata.ANGLE_SD), np.random.normal(0, pdata.NORM_SD)])
                    action = action + noise
                    if(abs(action[0]) > pdata.MAX_MOTOR_ACTION[0]):
                        action[0] = pdata.MAX_MOTOR_ACTION[0] if action[0] > pdata.MAX_MOTOR_ACTION[0] else -pdata.MAX_MOTOR_ACTION[0]
                    if(abs(action[1]) > pdata.MAX_MOTOR_ACTION[1]):
                        action[1] = pdata.MAX_MOTOR_ACTION[1] if action[1] > pdata.MAX_MOTOR_ACTION[1] else -pdata.MAX_MOTOR_ACTION[1]

                    next_state, reward, done = self.roads.step(self.agent, action)
                    next_state_her = np.concatenate([next_state, final_goal], axis=0) 
                    self.agent.add_to_replaybuffer(state_her, next_state_her, action, reward, np.float(done))

                    # add future to replay buffer
                    future_pos[0], future_pos[1] = next_state[14], next_state[15]
                    state_her = np.concatenate([state, future_pos], axis = 0)
                    next_state_her = np.concatenate([next_state, future_pos], axis = 0)
                    reward_her = self.roads.get_her_reward(self.agent, reward, future_pos)
                    self.agent.add_to_replaybuffer(state_her, next_state_her, action, reward_her, np.float(done))

                    state = next_state
                    if done or t >= pdata.MAX_LENGTH_OF_TRAJECTORY:
                        break
                self.logger.add_to_position_seq()

                if i % pdata.LOG_INTERVAL== 0:
                    # print('motor_thread_HER saves parameters.') # breathing messages
                    self.agent.save()

                # 样本池满了之后每轮迭代都在更新模型
                if len(self.agent.get_buffer_storage()) >= pdata.CAPACITY - 1:
                    # self.logger.write_to_log('HER params update!\n')
                    self.agent.update_model()
        self.logger.record()


# class motor_train_process(multiprocessing.Process):
class motor_train_process(threading.Thread):
    def __init__(self, origin_idx, veer_idx, agent_idx):
        super().__init__()
        # self.daemon = True
        self._origin_idx = origin_idx
        self._veer_idx = veer_idx
        mark_str = pdata.AGENT_TUPLE[agent_idx] + '_' + pdata.DIRECTION_TUPLE[origin_idx]+'_'+pdata.VEER_TUPLE[veer_idx]
        self.logger = log_util.logWriter(mark_str)
        self.roads = env.TrainingEnvironment(self.logger)

        if agent_idx == 0:
            self.agent = motorVehicle(self.logger)
        elif agent_idx == 1:
            self.agent = Bicycle(self.logger)
        # elif vehicle_str == pdata.AGENT_TUPLE[2]:
        #     self.agent == pedestrian()

    # 单agent的线程
    def run(self):
        # origin_idx: 小车的出发地   veer_idx 小车的转向
        self.agent.initiate_agent(pdata.DIRECTION_TUPLE[self._origin_idx], pdata.VEER_TUPLE[self._veer_idx])    
        
        # 收集训练数据
        if pdata.MODE == 'train':
            # self.logger.record_start_time()
            if pdata.LOAD: 
                self.agent.load()     # 决定了是否导入已有的模型

            for i in range(pdata.MAX_EPISODE):
                state = self.roads.reset(self.agent)      # state 的维数要尤其注意

                # count(): an infinite iterator provided by module itertools
                # 用于往经验池中放入样本的内层循环
                for t in count():
                    action = self.agent.select_action(state)

                    # add noise to action 
                    noise = np.array([np.random.normal(0, pdata.ANGLE_SD), np.random.normal(0, pdata.NORM_SD)])
                    action = action + noise
                    if(abs(action[0]) > pdata.MAX_MOTOR_ACTION[0]):
                        action[0] = pdata.MAX_MOTOR_ACTION[0] if action[0] > pdata.MAX_MOTOR_ACTION[0] else -pdata.MAX_MOTOR_ACTION[0]
                    if(abs(action[1]) > pdata.MAX_MOTOR_ACTION[1]):
                        action[1] = pdata.MAX_MOTOR_ACTION[1] if action[1] > pdata.MAX_MOTOR_ACTION[1] else -pdata.MAX_MOTOR_ACTION[1]

                    next_state, reward, done = self.roads.step(self.agent, action)
                    # if pdata.RENDER and i >= pdata.RENDER_INTERVAL: roads.render()

                    # the five-element tuple of a sample
                    # 往经验池里增加序列元素
                    self.agent.add_to_replaybuffer(state, next_state, action, reward, np.float(done))    # np.float(done)保存入元组是为了用于update计算中的数值转换

                    state = next_state
                    if done or t >= pdata.MAX_LENGTH_OF_TRAJECTORY:
                        break
                self.logger.add_to_position_seq()

                if i % pdata.LOG_INTERVAL== 0:
                    print('motor_thread saves parameters.')
                    self.agent.save()

                # 样本池满了之后每轮迭代都在更新模型
                if len(self.agent.get_buffer_storage()) >= pdata.CAPACITY - 1:
                    # self.logger.write_to_log('params update!\n')
                    self.agent.update_model()  # 更新
        self.logger.record()   


if __name__ == '__main__':
    # 初始化数据
    print('parent process start')
    # p_straight = motor_train_process(0, 0, 0)  # 自西往东
    p_left_1 = motor_train_process(2,1,0)  # 自东往南
    p_left_2 = motor_train_process(3,1,0) # 自北往东
    # p_straight_her = motor_train_process_HER(0, 0, 0) #自西往东

    # p_straight.start()
    p_left_1.start()
    p_left_2.start()
    # p_straight_her.start()

    # p_straight.join()
    p_left_1.join()
    p_left_2.join()
    # p_straight_her.join()
    print('parent process ends.')

