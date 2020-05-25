import random, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import public_data as pdata
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 行动器 —— 继承自 nn.Module，则必须覆盖的函数有 __init__ 和 forward
# 在Actor-Critic 框架中， Actor用于输出动作（不管是输出概率性的离散动作，还是动作以策略网络的形式输出）
# 因此在 Actor网络中，输入是 state-vector ，输出是 action-vector
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        # max_action: 用于限制神经网络的输出，既然与 Tensor对象一起作为操作数，则必须保证 max_action 也为Tensor
        super(Actor, self).__init__()   # 这里的写法是为了调用父类的构造函数

        """ 三层的全连接神经网络 —— 真正神经网络的部分反而很简单 """
        self.l1 = nn.Linear(state_dim, 100)
        self.l2 = nn.Linear(100, 75)
        self.l3 = nn.Linear(75, action_dim)
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
        # max_action : 用来反归一化
        x = self.max_action * torch.tanh(x)
        return x


class ActorHER(nn.Module):
    def __init__(self, her_state_dim, action_dim, max_action):
        super(ActorHER, self).__init__() 
        self.l1 = nn.Linear(her_state_dim, 100) # eev_state + pos_goal 的拼接维度
        self.l2 = nn.Linear(100, 75)
        self.l3 = nn.Linear(75, action_dim)
        self.max_action = torch.FloatTensor(max_action.reshape(1, -1)).to(device) 

    def forward(self, x):
        x = F.relu(self.l1(x))  
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = self.max_action * torch.tanh(x)
        return x


class ActorPedestrian(nn.Module):
    def __init__(self, state_dim, action_dim, max_vel):
        super(ActorPedestrian, self).__init__()
        self.l1 = nn.Linear(state_dim, 50)
        self.l2 = nn.Linear(50, 30)
        self.l3 = nn.Linear(30, action_dim)
        self.max_velocity = max_vel

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)  # 一个速度矢量
        ratio = self.max_velocity / (torch.norm(x) + pdata.EPSILON)
        if ratio < 1:
            x = x * ratio
        return x


class ActorPedestrianHER(nn.Module):
    def __init__(self, her_state_dim, action_dim, max_vel):
        super(ActorPedestrianHER, self).__init__() 
        self.l1 = nn.Linear(her_state_dim, 50) # env_state + pos_goal 的拼接维度
        self.l2 = nn.Linear(50, 30)
        self.l3 = nn.Linear(30, action_dim)
        self.max_velocity = max_vel

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)  # 一个速度矢量
        ratio = self.max_velocity / (torch.norm(x) + pdata.EPSILON)
        if ratio < 1:
            x = x * ratio
        return x


# 评价器 ：作用就是输出 Q(s, a) 的估计
# 也因此，在 Actor-Critic 框架中，Critic的输入元一定是 state-vector with action-vector
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        """ 还是三层的全连接神经网络 """ 
        ''' what's the meaning of actions were not included until the hidden layer of Q '''
        self.l1 = nn.Linear(state_dim + action_dim, 100)    # 输入是 state 和 action 的重组向量
        self.l2 = nn.Linear(100, 75)
        self.l3 = nn.Linear(75, 1)     # 没有激活函数 —— 因为Critic网络本身就是输出价值的，不需要归一化

    def forward(self, x, u):
        # x = F.relu(self.l1(torch.cat([x, u], 1)))   # torch.cat(seq, dim, out=None)
        x = F.relu(self.l1(torch.cat((x, u),1)))  # torch.cat() 只能 concatenate 相同 shape 的 tensor
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CriticHER(nn.Module):
    def __init__(self, her_state_dim, action_dim):
        super(CriticHER, self).__init__()
        self.l1 = nn.Linear(her_state_dim + action_dim, 100)    # 输入是 state，position 和 action 的重组向量
        self.l2 = nn.Linear(100, 75)
        self.l3 = nn.Linear(75, 1)

    # x 和 u 是一个 minibatch 的向量
    def forward(self, x, a):
        x = F.relu(self.l1(torch.cat((x, a),dim=1)))  # x: united_state_batch , a: united_action_batch
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CriticPedestrian(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticPedestrian, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 50)    # 输入是 state，position 和 action 的重组向量
        self.l2 = nn.Linear(50, 30)
        self.l3 = nn.Linear(30, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat((x, u),1)))  # x: state||goal , u: action
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CriticPedestrianHER(nn.Module):
    def __init__(self, her_state_dim, action_dim):
        super(CriticPedestrianHER, self).__init__()
        self.l1 = nn.Linear(her_state_dim + action_dim, 50)
        self.l2 = nn.Linear(50, 30)
        self.l3 = nn.Linear(30, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat((x, u),1)))  # torch.cat() 只能 concatenate 相同 shape 的 tensor
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# 用于 MADDPG 算法的 Critic网络
class CriticCentral(nn.Module):
    def __init__(self, agent_n):
        super(CriticCentral, self).__init__()
        self.input_dimenstion = (pdata.STATE_DIMENSION + pdata.ACTION_DIMENSION) * agent_n
        self.l1 = nn.Linear(self.input_dimenstion, 100)    # 输入是 state 和 action 的重组向量
        self.l2 = nn.Linear(100, 75)
        self.l3 = nn.Linear(75, 1)     # 没有激活函数 —— 因为Critic网络本身就是输出价值的，不需要归一化

    def forward(self, obs, acts):
        combined = torch.cat((obs, acts), 1)
        x = F.relu(self.l1(combined))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CriticCentralPE(nn.Module):
    def __init__(self, agent_n):
        super(CriticCentralPE, self).__init__()
        self.input_dimenstion = (pdata.STATE_DIMENSION + pdata.ACTION_DIMENSION) * agent_n
        self.l1 = nn.Linear(self.input_dimenstion, 50)
        self.l2 = nn.Linear(50, 30)
        self.l3 = nn.Linear(30, 1) 

    def forward(self, obs, acts):
        combined = torch.cat((obs, acts), 1)
        x = F.relu(self.l1(combined))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
