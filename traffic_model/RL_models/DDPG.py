from itertools import count

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

import public_data as pdata  #相对路径问题

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=pdata.CAPACITY):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            # 经验回放池的超限更新策略
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size       #循环数组更新
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)  # 在 0 到 len(self.storage) 之间产生 size 个数
        x, y, u, r, d = [], [], [], [], []

        # 这里的样本是完全随机的64个
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Replay_buffer_HER():
    def __init__(self, max_size=pdata.CAPACITY):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            # 经验回放池的超限更新策略
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size       #循环数组更新
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size= pdata.HER_K)  # 在 0 到 len(self.storage) 之间产生 size 个数
        x, y, u, r, d = [], [], [], [], []

        # 这里的样本是完全随机的batch_size个
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            X_HER, Y_HER, U_HER, R_HER, D_HER = self.storage[i+1]
            x.append(np.array(X, copy=False)), x.append(np.array(X_HER, copy=False))
            y.append(np.array(Y, copy=False)), y.append(np.array(Y_HER, copy=False))
            u.append(np.array(U, copy=False)), u.append(np.array(U_HER, copy=False))
            r.append(np.array(R, copy=False)), r.append(np.array(R_HER, copy=False))
            d.append(np.array(D, copy=False)), d.append(np.array(D_HER, copy=False))

        for i in range(2*pdata.HER_K, batch_size):
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


# 行动器 —— 继承自 nn.Module，则必须覆盖的函数有 __init__ 和 forward
# 在Actor-Critic 框架中， Actor用于输出动作（不管是输出概率性的离散动作，还是动作以策略网络的形式输出）
# 因此在 Actor网络中，输入是 state-vector ，输出是 action-vector
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        # max_action: 用于限制神经网络的输出，既然与 Tensor对象一起作为操作数，则必须保证 max_action 也为Tensor
        super(Actor, self).__init__()   # 这里的写法是为了调用父类的构造函数

        """ 三层的全连接神经网络 —— 真正神经网络的部分反而很简单 """
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = torch.FloatTensor(max_action.reshape(1, -1)).to(device) 

    # forward 是神经网络每一次调用时用来计算输出的 —— 必须保证输入是个 torch.Tensor
    def forward(self, x):
        # self.l1 对象为 nn.Linear对象
        # input shape: (N, *, in_features) where * means any number of additional dimensions
        # output shape: (N, *, out_features) where all but the last dimension are the same shape as the input
        x = F.relu(self.l1(x))  
        x = F.relu(self.l2(x))
        x = self.l3(x)
        # torch.tanh() 利用双曲正切函数对输出tensor中的每一个元素进行了tanh的归一化计算
        # max_action : 用来限制输出每个维度范围的向量
        x = self.max_action * torch.tanh(x)
        return x


class ActorHER(nn.Module):
    def __init__(self, her_state_dim, action_dim, max_action):
        super(ActorHER, self).__init__() 
        self.l1 = nn.Linear(her_state_dim, 400) # eev_state + pos_goal 的拼接维度
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = torch.FloatTensor(max_action.reshape(1, -1)).to(device) 

    def forward(self, x):
        x = F.relu(self.l1(x))  
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = self.max_action * torch.tanh(x)
        return x


# 评价器 ：作用就是输出 Q(s, a) 的估计
# 也因此，在 Actor-Critic 框架中，Critic的输入元一定是 state-vector with action-vector
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        """ 还是三层的全连接神经网络 """ 
        ''' what's the meaning of actions were not included until the hidden layer of Q '''
        self.l1 = nn.Linear(state_dim + action_dim, 400)    # 输入是 state 和 action 的重组向量
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)     # 没有激活函数 —— 因为Critic网络本身就是输出价值的，不需要归一化

    def forward(self, x, u):
        # x = F.relu(self.l1(torch.cat([x, u], 1)))   # torch.cat(seq, dim, out=None)
        x = F.relu(self.l1(torch.cat((x, u),1)))  # torch.cat() 只能 concatenate 相同 shape 的 tensor
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CriticHER(nn.Module):
    def __init__(self, her_state_dim, action_dim):
        super(CriticHER, self).__init__()
        self.l1 = nn.Linear(her_state_dim + action_dim, 400)    # 输入是 state，position 和 action 的重组向量
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat((x, u),1)))  # x: state||goal , u: action
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# 一个 DDPG 实例对应一个 agent
# TODO：完成与自己项目相关的修改
class DDPG(object):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    veer = 'straight'
    origin = 'west'
    # state_dim : 状态维度
    # action_dim： 动作维度
    # max_action：动作限制向量
    def __init__(self, state_dim, action_dim, max_action, origin_str, veer_str, logger):

        # 存在于 GPU 的神经网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)    # origin_network
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)    # target_network
        self.actor_target.load_state_dict(self.actor.state_dict())  # initiate actor_target with actor's parameters
        # pytorch 中的 tensor 默认requires_grad 属性为false，即不参与梯度传播运算，特别地，opimizer中模型参数是会参与梯度优化的
        self.actor_optimizer = optim.Adam(self.actor.parameters(), pdata.LEARNING_RATE) # 以pdata.LEARNING_RATE指定学习率优化actor中的参数 

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), pdata.LEARNING_RATE)

        self.replay_buffer = Replay_buffer()    # initiate replay-buffer
        # self.writer = SummaryWriter(pdata.DIRECTORY)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.veer = veer_str
        self.origin = origin_str

        self.logger = logger


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # numpy.ndarray.flatten(): 返回一个 ndarray对象的copy，并且将该ndarray压缩成一维数组
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


    # update the parameters in actor network and critic network
    # 只有 replay_buffer 中的 storage 超过了样本数量才会调用 update函数
    def update(self):

        for it in range(pdata.UPDATE_ITERATION):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(pdata.BATCH_SIZE)     # 随机获取 batch_size 个五元组样本(sample random minibatch)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value —— Q(S', A') is an value evaluated with next_state and predicted action
            # 这里的 target_Q 是 sample 个 一维tensor
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            # detach(): Return a new tensor, detached from the current graph
            # soft update of target_Q
            target_Q = reward + ((1 - done) * pdata.GAMMA * target_Q).detach() 

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss : a mean-square error
            # 由论文，critic_loss 其实计算的是每个样本估计值与每个critic网络输出的均值方差
            # torch.nn.functional.mse_loss 为计算tensor中各个元素的的均值方差
            critic_loss = F.mse_loss(current_Q, target_Q) 
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            self.logger.write_to_log('critic_loss:{loss}'.format(loss=critic_loss))

            # Optimize the critic
            self.critic_optimizer.zero_grad()   # zeros the gradient buffer
            critic_loss.backward()              # back propagation on a dynamic graph
            self.critic_optimizer.step()

            # Compute actor loss
            # actor_loss：见论文中对公式 (6) 的理解
            # mean()：对tensor对象求所有element的均值
            # backward() 以梯度下降的方式更新参数，则将 actor_loss 设置为反向梯度，这样参数便往梯度上升方向更新
            actor_loss = -self.critic(state, self.actor(state)).mean()  
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            self.logger.write_to_log('actor_loss:{loss}'.format(loss=actor_loss))

            # Optimize the actor
            self.actor_optimizer.zero_grad()    # Clears the gradients of all optimized torch.Tensor
            actor_loss.backward()
            self.actor_optimizer.step()     # perform a single optimization step

            # 这里是两个 target网络的 soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(pdata.TAU * param.data + (1 - pdata.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(pdata.TAU * param.data + (1 - pdata.TAU) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1


    def save(self, agent_type):
        torch.save(self.actor.state_dict(), pdata.DIRECTORY + agent_type + '_actor_'+ self.veer + '.pth')
        torch.save(self.critic.state_dict(), pdata.DIRECTORY +  agent_type + '_critic_' + self.veer + '.pth')

    def load(self, agent_type):
        file_actor = pdata.DIRECTORY +  agent_type + '_actor_'+ self.veer + '.pth'
        file_critic = pdata.DIRECTORY +  agent_type + '_critic_'+ self.veer + '.pth'
        if os.path.exists(file_actor) and os.path.exists(file_critic):
            self.actor.load_state_dict(torch.load(file_actor))
            self.critic.load_state_dict(torch.load(file_critic))
        else:
            self.logger.write_to_log(".pth doesn't exist. Use default parameters")



class DDPG_HER(DDPG):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    veer = 'straight'
    origin = 'west'
    # state_dim : 状态维度
    # action_dim： 动作维度
    # max_action：动作限制向量
    def __init__(self, her_state_dim, action_dim, max_action, origin_str, veer_str, logger):

        # 存在于 GPU 的神经网络
        self.actor = ActorHER(her_state_dim, action_dim, max_action).to(device)    # origin_network
        self.actor_target = ActorHER(her_state_dim, action_dim, max_action).to(device)    # target_network
        self.actor_target.load_state_dict(self.actor.state_dict())  # initiate actor_target with actor's parameters

        self.actor_optimizer = optim.Adam(self.actor.parameters(), pdata.LEARNING_RATE) # 以pdata.LEARNING_RATE指定学习率优化actor中的参数 

        self.critic = CriticHER(her_state_dim, action_dim).to(device)
        self.critic_target = CriticHER(her_state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), pdata.LEARNING_RATE)

        self.replay_buffer = Replay_buffer_HER()    # initiate replay-buffer
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.veer = veer_str
        self.origin = origin_str

        self.logger = logger


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # numpy.ndarray.flatten(): 返回一个 ndarray对象的copy，并且将该ndarray压缩成一维数组
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


    # update the parameters in actor network and critic network
    # 只有 replay_buffer 中的 storage 超过了样本数量才会调用 update函数
    def update(self):
        # 更新次数
        for it in range(pdata.UPDATE_ITERATION):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(pdata.BATCH_SIZE)     # 随机获取 batch_size 个五元组样本(sample random minibatch)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * pdata.GAMMA * target_Q).detach() 

            current_Q = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q, target_Q) 
            self.logger.write_to_log('critic_loss:{loss}'.format(loss=critic_loss))

            # Optimize the critic
            self.critic_optimizer.zero_grad()   # zeros the gradient buffer
            critic_loss.backward()              # back propagation on a dynamic graph
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, self.actor(state)).mean()  
            self.logger.write_to_log('actor_loss:{loss}'.format(loss=actor_loss))

            # Optimize the actor
            self.actor_optimizer.zero_grad()    # Clears the gradients of all optimized torch.Tensor
            actor_loss.backward()
            self.actor_optimizer.step()     # perform a single optimization step

            # 这里是两个 target网络的 soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(pdata.TAU * param.data + (1 - pdata.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(pdata.TAU * param.data + (1 - pdata.TAU) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1


    def save(self, agent_type):
        torch.save(self.actor.state_dict(), pdata.DIRECTORY + agent_type + '_actor_HER'+ self.veer + '.pth')
        torch.save(self.critic.state_dict(), pdata.DIRECTORY + agent_type + '_critic_HER'+ self.veer + '.pth')

    def load(self, agent_type):
        file_actor = pdata.DIRECTORY +  agent_type + '_actor_HER'+ self.veer + '.pth'
        file_critic = pdata.DIRECTORY +  agent_type + '_critic_HER'+ self.veer + '.pth'
        if os.path.exists(file_actor) and os.path.exists(file_critic):
            self.actor.load_state_dict(torch.load(file_actor))
            self.critic.load_state_dict(torch.load(file_critic))
        else:
            self.logger.write_to_log(".pth doesn't exist. Use default parameters")