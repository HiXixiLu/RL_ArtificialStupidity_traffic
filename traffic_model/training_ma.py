from itertools import count
import os, sys, random, copy 
# 在sys.path里动态添加了路径后，python才能找到项目中的模块
# 坑：在linux里使用 python 命令运行该.py 时，os.getcwd() 会随着用户当前所处的目录位置改变
sys.path.append(os.getcwd() + '/traffic_model/RL_models') 
import multiprocessing,threading
from threading import Timer
import json
import numpy as np
from numpy import linalg as la
import torch
from RL_models import environment as env
from RL_models import log_util as lo
from RL_models.replaybuffer import CentralReplayBufferMA
from RL_models import public_data as pdata
from RL_models.vehicle import MotorVehicleMA, BicycleMA
from RL_models.pedestrian import PedestrianMA


if __name__ == '__main__':
    ma_env = env.EnvironmentMA()
    # agent initiation:
    ag_m = MotorVehicleMA(lo.logWriter('MotorVehicleMA_w_e'))
    ag_m.initiate(np.array([-pdata.MOTER_L-pdata.LANE_W, -pdata.LANE_W/2]),
    np.array([pdata.LANE_W + pdata.MOTER_L, -pdata.LANE_W/2]), 
    np.array([pdata.MAX_VELOCITY / 5, 0.0]),
    'w_e', 3)

    ag_b = BicycleMA(lo.logWriter('BicycleMA_e_s'))
    ag_b.initiate(np.array([pdata.LANE_W + pdata.NON_MOTOR_L, pdata.LANE_W /2]),
    np.array([-pdata.LANE_W, -pdata.NON_MOTOR_L - pdata.LANE_W]),
    np.array([-pdata.MAX_BICYCLE_VEL/5, 0.0]),
    'e_s', 3)

    ag_p = PedestrianMA(lo.logWriter('PedestrianMA'))
    ag_p.initiate(1, 2.0, 3)

    ma_env.join_agent([ag_m, ag_b, ag_p])

    central_replay_buffer = CentralReplayBufferMA()

    ma_env.reset()
    united_state = ma_env.get_united_state_feature()
    united_next_state = copy.deepcopy(united_state)

    for i in range(pdata.MAX_EPISODE):
        united_state = united_next_state
        actions = [ma_env.agent_queue[j].select_action(united_state[j*pdata.STATE_DIMENSION : j*pdata.STATE_DIMENSION + pdata.STATE_DIMENSION])
                    for j in range(len(ma_env.agent_queue))]

        united_next_state, reward, done = ma_env.step(actions)

        for k in range(len(ma_env.agent_queue)):
            start = k * pdata.ACTION_DIMENSION
            ma_env.agent_queue[k].normalize_action(actions[start: start + pdata.ACTION_DIMENSION])

        actions = np.array(actions)
        actions = actions.flatten()
        central_replay_buffer.push((united_state, united_next_state, actions, reward, np.float(done)))

        if pdata.CAPACITY <= len(central_replay_buffer.storage):
            ma_env.update_policy(central_replay_buffer)

        if i % pdata.LOG_INTERVAL== 0:
            # print('motor_thread saves parameters.')
            for ag in ma_env.agent_queue:
                ag.save()

    for ag in ma_env.agent_queue:
        ag.logger.record()
