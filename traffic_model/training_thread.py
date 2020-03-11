from itertools import count
import os, sys, random,time  
sys.path.append(os.getcwd() + '\\traffic_model\\RL_models')     # 在sys.path里动态添加了路径后，python才能找到项目中的模块
import json
import numpy as np
import torch
from RL_models import environment as env
from RL_models import DDPG
from RL_models import public_data as pdata
from RL_models.motor import MotorVehicle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
# env = gym.make(args.env_name).unwrapped
roads = env.IntersectionEnvironment()

min_Val = torch.tensor(1e-7).float().to(device) # min value : .to(device) 表示由 GPU 训练


def training_main_thread():
 
    # 如果要训练多个 agent 是否就要用一个集合保存 agent（同一线程）？ 还是多线程训练 agent？
    # agent = DDPG.DDPG_Straight(pdata.STATE_DIMENSION, pdata.ACTION_DIMENSION, pdata.MAX_MOTOR_ACTION)
    agent = MotorVehicle()
    pdata.record_hard_params()
    # agent的初始化
    agent.initiate_agent("west",'straight')
    
    ep_r = 0
    if pdata.MODE == 'test':
        agent.load()    # 只有一个 agent
        for i in range(pdata.TEST_ITERATION):
            state = roads.reset(agent)   # 该初始状态很重要
            # itertools.count(n) : 若忽略n，则从 0 开始生成整数序列
            # count() 保证了序列的长度可以是不定的
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done = roads.step(agent, action)
                ep_r += reward
                # roads.render()
                # 由于有该句的 if done or t >=...  作为条件控制跳出，则不需要控制外层 for t in count() 的循环
                if done or t >= pdata.MAX_LENGTH_OF_TRAJECTORY:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state
    # 收集训练数据
    elif pdata.MODE == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        pdata.write_to_log('START TIME: {start}'.format(start=time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime(time.time()))))
        if pdata.LOAD: 
            agent.load()     # 决定了是否导入已有的模型

        for i in range(pdata.MAX_EPISODE):
            state = roads.reset(agent)      # state 的维数要尤其注意

            # count(): an infinite iterator provided by module itertools
            # 用于往经验池中放入样本的内层循环
            for t in count():
                action = agent.select_action(state)

                # issue 3 add noise to action 
                action = (action + np.random.normal(0, pdata.EXPLORATION_NOISE, size=pdata.ACTION_DIMENSION))
                if(abs(action[0]) > pdata.MAX_MOTOR_ACTION[0]):
                    action[0] = pdata.MAX_MOTOR_ACTION[0] if action[0] > pdata.MAX_MOTOR_ACTION[0] else -pdata.MAX_MOTOR_ACTION[0]
                if(abs(action[1]) > pdata.MAX_MOTOR_ACTION[1]):
                    action[1] = pdata.MAX_MOTOR_ACTION[1] if action[1] > pdata.MAX_MOTOR_ACTION[1] else -pdata.MAX_MOTOR_ACTION[1]

                next_state, reward, done = roads.step(agent, action)
                ep_r += reward
                # if pdata.RENDER and i >= pdata.RENDER_INTERVAL: roads.render()

                # the five-element tuple of a sample
                # 往经验池里增加序列元素
                agent.add_to_replaybuffer(state, next_state, action, reward, np.float(done))    # np.float(done)保存入元组是为了用于update计算中的数值转换
                if (i+1) % 10 == 0:
                    print('---------  Episode {},  The memory size is {}  --------'.format(i, len(agent.get_buffer_storage())))

                state = next_state
                if done or t >= pdata.MAX_LENGTH_OF_TRAJECTORY:
                    # agent.writer.add_scalar('ep_r', ep_r, global_step=i) 
                    if i % pdata.PRINT_LOG== 0:
                        print("------ Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{} --------".format(i, ep_r, t))
                        print("------ Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{} --------".format(i, ep_r, t), file = pdata.EXPERIMENT_LOG)
                    ep_r = 0
                    break

            if i % pdata.LOG_INTERVAL== 0:
                agent.save()

            # 样本池满了之后每轮迭代都在更新模型
            if len(agent.get_buffer_storage()) >= pdata.CAPACITY - 1:
                pdata.write_to_log('params update!\n')
                agent.update_model()  # 更新

        pdata.write_to_log('END TIME: {end}'.format(end=time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime(time.time()))))
        pdata.close_file() 

    else:
        pdata.write_to_log('training_thread: mode wrong.')
        pdata.close_file()
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    # 初始化数据
    training_main_thread()
    