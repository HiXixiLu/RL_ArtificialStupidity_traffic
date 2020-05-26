from itertools import count
import random, copy, os
import numpy as np
import numpy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import public_data as pdata  #相对路径问题
from network import Actor, ActorHER, ActorPedestrian, ActorPedestrianHER,Critic,CriticHER,CriticPedestrian,CriticPedestrianHER,CriticCentral,CriticCentralPE
from replaybuffer import Replay_buffer, Replay_buffer_HER, FilterReplayBuffer

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !  
'''

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
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)    # origin_network
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)    # target_network
        self.actor_target.load_state_dict(self.actor.state_dict())  # initiate actor_target with actor's parameters
        # pytorch 中的 tensor 默认requires_grad 属性为false，即不参与梯度传播运算，特别地，opimizer中模型参数是会参与梯度优化的
        self.actor_optimizer = optim.Adam(self.actor.parameters(), pdata.LEARNING_RATE) # 以pdata.LEARNING_RATE指定学习率优化actor中的参数 

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), pdata.LEARNING_RATE)

        # self.replay_buffer = Replay_buffer()    # initiate replay-buffer
        self.replay_buffer = FilterReplayBuffer()
        self.writer = SummaryWriter(pdata.DIRECTORY+'runs')
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.veer = veer_str
        self.origin = origin_str

        self.logger = logger


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # numpy.ndarray.flatten(): 返回一个 ndarray对象的copy，并且将该ndarray压缩成一维数组
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


    # update the parameters in actor network and critic network
    # 只有 replay_buffer 中的 storage 超过了样本数量才会调用 update函数
    def update(self):
        critic_loss_list = []
        actor_performance_list = []
        for it in range(pdata.UPDATE_ITERATION):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(pdata.BATCH_SIZE)     # 随机获取 batch_size 个五元组样本(sample random minibatch)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value —— Q(S', A') is an value evaluated with next_state and predicted action
            # 这里的 target_Q 是 sample 个 一维tensor
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            # detach(): Return a new tensor, detached from the current graph
            # evaluate Q: targetQ = R + γQ'(s', a')
            target_Q = reward + ((1 - done) * pdata.GAMMA * target_Q).detach()  # batch_size个维度

            # Get current Q estimate
            current_Q = self.critic(state, action)  # 1 维 

            # Compute critic loss : a mean-square error
            # 由论文，critic_loss 其实计算的是每个样本估计值与每个critic网络输出的均值方差
            # torch.nn.functional.mse_loss 为计算tensor中各个元素的的均值方差
            critic_loss = F.mse_loss(current_Q, target_Q) 
            self.writer.add_scalar('critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # self.logger.write_to_log('critic_loss:{loss}'.format(loss=critic_loss))
            # self.logger.add_to_critic_buffer(critic_loss.item())
            critic_loss_list.append(critic_loss.item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()   # zeros the gradient buffer
            critic_loss.backward()              # back propagation on a dynamic graph
            self.critic_optimizer.step()

            # Compute actor loss
            # actor_loss：见论文中对公式 (6) 的理解
            # mean()：对tensor对象求所有element的均值
            # backward() 以梯度下降的方式更新参数，则将 actor_loss 设置为反向梯度，这样参数便往梯度上升方向更新
            actor_loss = -self.critic(state, self.actor(state)).mean()  
            self.writer.add_scalar('actor_performance', actor_loss, global_step=self.num_actor_update_iteration)
            # self.logger.write_to_log('actor_loss:{loss}'.format(loss=actor_loss))
            # self.logger.add_to_actor_buffer(actor_loss.item())
            actor_performance_list.append(actor_loss.item())

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

        actor_performance = np.mean(np.array(actor_performance_list)).item()
        self.logger.add_to_actor_buffer(actor_performance)
        critic_loss = np.mean(critic_loss_list).item()
        self.logger.add_to_critic_buffer(critic_loss)


    def save(self, mark_str):
        torch.save(self.actor.state_dict(), pdata.DIRECTORY + mark_str + '_actor.pth')
        torch.save(self.critic.state_dict(), pdata.DIRECTORY +  mark_str + '_critic.pth')
        

    def load(self, mark_str):
        file_actor = pdata.DIRECTORY +  mark_str + '_actor.pth'
        file_critic = pdata.DIRECTORY +  mark_str + '_critic.pth'
        if os.path.exists(file_actor) and os.path.exists(file_critic):
            self.actor.load_state_dict(torch.load(file_actor))
            self.critic.load_state_dict(torch.load(file_critic))
        # else:
            # self.logger.write_to_log(".pth doesn't exist. Use default parameters")


class DDPG_HER(DDPG):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    veer = 'straight'
    origin = 'west'
    # state_dim : 状态维度
    # action_dim： 动作维度
    # max_action：动作限制向量
    def __init__(self, her_state_dim, action_dim, max_action, origin_str, veer_str, logger):

        # 存在于 GPU 的神经网络
        self.actor = ActorHER(her_state_dim, action_dim, max_action).to(self.device)    # origin_network
        self.actor_target = ActorHER(her_state_dim, action_dim, max_action).to(self.device)    # target_network
        self.actor_target.load_state_dict(self.actor.state_dict())  # initiate actor_target with actor's parameters

        self.actor_optimizer = optim.Adam(self.actor.parameters(), pdata.LEARNING_RATE) # 以pdata.LEARNING_RATE指定学习率优化actor中的参数 

        self.critic = CriticHER(her_state_dim, action_dim).to(self.device)
        self.critic_target = CriticHER(her_state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), pdata.LEARNING_RATE)

        self.replay_buffer = Replay_buffer_HER()    # initiate replay-buffer
        self.writer = SummaryWriter(pdata.DIRECTORY+'runs')
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.veer = veer_str
        self.origin = origin_str

        self.logger = logger


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # numpy.ndarray.flatten(): 返回一个 ndarray对象的copy，并且将该ndarray压缩成一维数组
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


    # update the parameters in actor network and critic network
    # 只有 replay_buffer 中的 storage 超过了样本数量才会调用 update函数
    def update(self):
        critic_loss_list = []
        actor_performance_list = []
        # 更新次数
        for it in range(pdata.UPDATE_ITERATION):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(pdata.BATCH_SIZE)     # 随机获取 batch_size 个五元组样本(sample random minibatch)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * pdata.GAMMA * target_Q).detach() 

            current_Q = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q, target_Q) 
            self.writer.add_scalar('HER_critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # self.logger.write_to_log('critic_loss:{loss}'.format(loss=critic_loss))
            # self.logger.add_to_critic_buffer(critic_loss.item())
            critic_loss_list.append(critic_loss.item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()   # zeros the gradient buffer
            critic_loss.backward()              # back propagation on a dynamic graph
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, self.actor(state)).mean()  
            self.writer.add_scalar('HER_actor_performance', actor_loss, global_step=self.num_actor_update_iteration)
            # self.logger.write_to_log('actor_loss:{loss}'.format(loss=actor_loss))
            # self.logger.add_to_actor_buffer(actor_loss.item())
            actor_performance_list.append(actor_loss.item())

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

        actor_performance = np.mean(np.array(actor_performance_list)).item()
        self.logger.add_to_actor_buffer(actor_performance)
        critic_loss = np.mean(critic_loss_list).item()
        self.logger.add_to_critic_buffer(critic_loss)


    def save(self, mark_str):
        torch.save(self.actor.state_dict(), pdata.DIRECTORY + mark_str + '_actor_HER.pth')
        torch.save(self.critic.state_dict(), pdata.DIRECTORY + mark_str + '_critic_HER.pth')

    def load(self, mark_str):
        file_actor = pdata.DIRECTORY +  mark_str + '_actor_HER.pth'
        file_critic = pdata.DIRECTORY +  mark_str + '_critic_HER.pth'
        if os.path.exists(file_actor) and os.path.exists(file_critic):
            self.actor.load_state_dict(torch.load(file_actor))
            self.critic.load_state_dict(torch.load(file_critic))
        # else:
        #     self.logger.write_to_log(".pth doesn't exist. Use default parameters")


class DDPG_PE(object):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # state_dim : 状态维度
    # action_dim： 动作维度
    # max_action：动作限制向量
    def __init__(self, state_dim, action_dim, max_velocity, logger):

        # 存在于 GPU 的神经网络
        self.actor = ActorPedestrian(state_dim, action_dim, max_velocity).to(self.device)    # origin_network
        self.actor_target = ActorPedestrian(state_dim, action_dim, max_velocity).to(self.device)    # target_network
        self.actor_target.load_state_dict(self.actor.state_dict())  # initiate actor_target with actor's parameters
        # pytorch 中的 tensor 默认requires_grad 属性为false，即不参与梯度传播运算，特别地，opimizer中模型参数是会参与梯度优化的
        self.actor_optimizer = optim.Adam(self.actor.parameters(), pdata.LEARNING_RATE) # 以pdata.LEARNING_RATE指定学习率优化actor中的参数 

        self.critic = CriticPedestrian(state_dim, action_dim).to(self.device)
        self.critic_target = CriticPedestrian(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), pdata.LEARNING_RATE)

        # self.replay_buffer = Replay_buffer()    # initiate replay-buffer
        self.replay_buffer = FilterReplayBuffer()
        self.writer = SummaryWriter(pdata.DIRECTORY+'runs')
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.logger = logger


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # numpy.ndarray.flatten(): 返回一个 ndarray对象的copy，并且将该ndarray压缩成一维数组
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


    # update the parameters in actor network and critic network
    # 只有 replay_buffer 中的 storage 超过了样本数量才会调用 update函数
    def update(self):
        critic_loss_list = []
        actor_performance_list = []
        for it in range(pdata.UPDATE_ITERATION):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(pdata.BATCH_SIZE)     # 随机获取 batch_size 个五元组样本(sample random minibatch)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

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
            self.writer.add_scalar('PE_critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # self.logger.write_to_log('critic_loss:{loss}'.format(loss=critic_loss))
            # self.logger.add_to_critic_buffer(critic_loss.item())
            critic_loss_list.append(critic_loss.item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()   # zeros the gradient buffer
            critic_loss.backward()              # back propagation on a dynamic graph
            self.critic_optimizer.step()

            # Compute actor loss
            # actor_loss：见论文中对公式 (6) 的理解
            # mean()：对tensor对象求所有element的均值
            # backward() 以梯度下降的方式更新参数，则将 actor_loss 设置为反向梯度，这样参数便往梯度上升方向更新
            actor_loss = -self.critic(state, self.actor(state)).mean()  
            self.writer.add_scalar('PE_actor_performance', actor_loss, global_step=self.num_actor_update_iteration)
            # self.logger.write_to_log('actor_loss:{loss}'.format(loss=actor_loss))
            # self.logger.add_to_actor_buffer(actor_loss.item())
            actor_performance_list.append(actor_loss.item())

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

        actor_performance = np.mean(np.array(actor_performance_list)).item()
        self.logger.add_to_actor_buffer(actor_performance)
        critic_loss = np.mean(critic_loss_list).item()
        self.logger.add_to_critic_buffer(critic_loss)


    def save(self, mark_str):
        torch.save(self.actor.state_dict(), pdata.DIRECTORY + mark_str + '_actor.pth')
        torch.save(self.critic.state_dict(), pdata.DIRECTORY +  mark_str + '_critic.pth')

    def load(self, mark_str):
        file_actor = pdata.DIRECTORY +  mark_str + '_actor.pth'
        file_critic = pdata.DIRECTORY +  mark_str + '_critic.pth'
        if os.path.exists(file_actor) and os.path.exists(file_critic):
            self.actor.load_state_dict(torch.load(file_actor))
            self.critic.load_state_dict(torch.load(file_critic))
        # else:
        #     self.logger.write_to_log(".pth doesn't exist. Use default parameters")


class DDPG_PE_HER(object):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # state_dim : 状态维度
    # action_dim： 动作维度
    # max_action：动作限制向量
    def __init__(self, her_state_dim, action_dim, max_velocity, logger):

        # 存在于 GPU 的神经网络
        self.actor = ActorPedestrianHER(her_state_dim, action_dim, max_velocity).to(self.device)    # origin_network
        self.actor_target = ActorPedestrianHER(her_state_dim, action_dim, max_velocity).to(self.device)    # target_network
        self.actor_target.load_state_dict(self.actor.state_dict())  # initiate actor_target with actor's parameters
        # pytorch 中的 tensor 默认requires_grad 属性为false，即不参与梯度传播运算，特别地，opimizer中模型参数是会参与梯度优化的
        self.actor_optimizer = optim.Adam(self.actor.parameters(), pdata.LEARNING_RATE) # 以pdata.LEARNING_RATE指定学习率优化actor中的参数 

        self.critic = CriticPedestrian(her_state_dim, action_dim).to(self.device)
        self.critic_target = CriticPedestrian(her_state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), pdata.LEARNING_RATE)

        self.replay_buffer = Replay_buffer_HER()    # initiate replay-buffer
        self.writer = SummaryWriter(pdata.DIRECTORY+'runs')
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.logger = logger


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # numpy.ndarray.flatten(): 返回一个 ndarray对象的copy，并且将该ndarray压缩成一维数组
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


    # update the parameters in actor network and critic network
    # 只有 replay_buffer 中的 storage 超过了样本数量才会调用 update函数
    def update(self):
        critic_loss_list = []
        actor_performance_list = []
        for it in range(pdata.UPDATE_ITERATION):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(pdata.BATCH_SIZE)     # 随机获取 batch_size 个五元组样本(sample random minibatch)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

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
            self.writer.add_scalar('PE_HER_critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # self.logger.write_to_log('critic_loss:{loss}'.format(loss=critic_loss))
            # self.logger.add_to_critic_buffer(critic_loss.item())
            critic_loss_list.append(critic_loss.item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()   # zeros the gradient buffer
            critic_loss.backward()              # back propagation on a dynamic graph
            self.critic_optimizer.step()

            # Compute actor loss
            # actor_loss：见论文中对公式 (6) 的理解
            # mean()：对tensor对象求所有element的均值
            # backward() 以梯度下降的方式更新参数，则将 actor_loss 设置为反向梯度，这样参数便往梯度上升方向更新
            actor_loss = -self.critic(state, self.actor(state)).mean()  
            self.writer.add_scalar('PE_HER_actor_performance', actor_loss, global_step=self.num_actor_update_iteration)
            # self.logger.write_to_log('actor_loss:{loss}'.format(loss=actor_loss))
            # self.logger.add_to_actor_buffer(actor_loss.item())
            actor_performance_list.append(actor_loss.item())

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

        actor_performance = np.mean(np.array(actor_performance_list)).item()
        self.logger.add_to_actor_buffer(actor_performance)
        critic_loss = np.mean(critic_loss_list).item()
        self.logger.add_to_critic_buffer(critic_loss)


    def save(self, mark_str):
        torch.save(self.actor.state_dict(), pdata.DIRECTORY + mark_str + '_actor_HER.pth')
        torch.save(self.critic.state_dict(), pdata.DIRECTORY +  mark_str + '_critic_HER.pth')

    def load(self, mark_str):
        file_actor = pdata.DIRECTORY +  mark_str + '_actor_HER.pth'
        file_critic = pdata.DIRECTORY +  mark_str + '_critic_HER.pth'
        if os.path.exists(file_actor) and os.path.exists(file_critic):
            self.actor.load_state_dict(torch.load(file_actor))
            self.critic.load_state_dict(torch.load(file_critic))
        # else:
        #     self.logger.write_to_log(".pth doesn't exist. Use default parameters")


class MADDPG(object):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__(self, state_dim, action_dim, max_action, agent_n, logger):
         # 存在于 GPU 的神经网络
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)    # origin_network
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)    # target_network
        self.actor_target.load_state_dict(self.actor.state_dict())  # initiate actor_target with actor's parameters
        # pytorch 中的 tensor 默认requires_grad 属性为false，即不参与梯度传播运算，特别地，opimizer中模型参数是会参与梯度优化的
        self.actor_optimizer = optim.Adam(self.actor.parameters(), pdata.LEARNING_RATE) # 以pdata.LEARNING_RATE指定学习率优化actor中的参数 

        self.critic = CriticCentral(agent_n).to(self.device)
        self.critic_target = CriticCentral(agent_n).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), pdata.LEARNING_RATE)
        # self.replay_buffer 取消：每个Agent不再有独立的经验池
        self.writer = SummaryWriter(pdata.DIRECTORY+'runs')
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.logger = logger


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # numpy.ndarray.flatten(): 返回一个 ndarray对象的copy，并且将该ndarray压缩成一维数组
        action = self.actor(state).cpu().data.numpy().flatten()
        return action       # 未归一化的

    def select_target_actions(self, states):
        # state 的输入必须是归一化的[[],[]] np.ndarray
        united_state = torch.FloatTensor(states).to(self.device)
        action = self.actor_target(united_state).cpu().data.numpy()
        return action   # 未归一化的

    def select_current_actions(self, states):
        united_state = torch.FloatTensor(states).to(self.device)
        action = self.actor(united_state).cpu().data.numpy()
        return action   # 未归一化的


    # update the parameters in actor network and critic network
    def update(self, central_replay_buffer_ma, agent_list, agent_idx):
        critic_loss_list = []
        actor_performance_list = []
        # Sample replay buffer:  [(united_normalized_states, united_normalized_next_states, united__normalized_action, [reward_1, ...], done), (...)]
        x, y, u, r, d = central_replay_buffer_ma.sample(pdata.BATCH_SIZE)     # 随机获取 batch_size 个五元组样本(sample random minibatch)
        # TODO: 从这里以下要改造成MADDPG的范式 
        next_actions = [agent_list[i].select_target_actions(y[:,i*pdata.STATE_DIMENSION : i*pdata.STATE_DIMENSION + pdata.STATE_DIMENSION]) 
                        for i in range(len(agent_list))]
        next_actions = np.concatenate(next_actions, axis=1)

        united_next_action_batch = torch.FloatTensor(next_actions).to(self.device)
        united_state_batch = torch.FloatTensor(x).to(self.device)   
        united_action_batch = torch.FloatTensor(u).to(self.device)  
        united_next_state_batch = torch.FloatTensor(y).to(self.device)  
        done = torch.FloatTensor(d).to(self.device) 
        reward_batch = torch.FloatTensor(r[:,agent_idx])  # torch.Size([64])
        reward_batch = reward_batch[:, None].to(self.device)    # torch.Size([64,1])
        target_Q = self.critic_target(united_next_state_batch, united_next_action_batch)    # torch.Size([64,1])
        target_Q = reward_batch + ((1 - done) * pdata.GAMMA * target_Q).detach()  # batch_size个维度
        current_Q = self.critic(united_state_batch, united_action_batch) 

        critic_loss = F.mse_loss(current_Q, target_Q) 
        self.writer.add_scalar('critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
        # self.logger.write_to_log('critic_loss:{loss}'.format(loss=critic_loss))
        # self.logger.add_to_critic_buffer(critic_loss.item())
        critic_loss_list.append(critic_loss.item())
        self.critic_optimizer.zero_grad()   # zeros the gradient buffer
        critic_loss.backward()              # back propagation on a dynamic graph
        self.critic_optimizer.step()

        current_actions = [agent_list[i].select_current_actions(x[:,i*pdata.STATE_DIMENSION : i*pdata.STATE_DIMENSION + pdata.STATE_DIMENSION])
                             for i in range(len(agent_list))]
        current_actions_batch = torch.FloatTensor(np.concatenate(current_actions, axis=1)).to(self.device)
        actor_loss = -self.critic(united_state_batch, current_actions_batch).mean()  
        self.writer.add_scalar('actor_performance', actor_loss, global_step=self.num_actor_update_iteration)
        # self.logger.write_to_log('actor_loss:{loss}'.format(loss=actor_loss))
        # self.logger.add_to_actor_buffer(actor_loss.item())
        actor_performance_list.append(actor_loss.item())

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

        actor_performance = np.mean(np.array(actor_performance_list)).item()
        self.logger.add_to_actor_buffer(actor_performance)
        critic_loss = np.mean(critic_loss_list).item()
        self.logger.add_to_critic_buffer(critic_loss)


    def save(self, mark_str):
        torch.save(self.actor.state_dict(), pdata.DIRECTORY + mark_str + '_actor_ma.pth')
        torch.save(self.critic.state_dict(), pdata.DIRECTORY +  mark_str + '_critic_ma.pth')
        

    def load(self, mark_str):
        file_actor = pdata.DIRECTORY +  mark_str + '_actor_ma.pth'
        file_critic = pdata.DIRECTORY +  mark_str + '_critic_ma.pth'
        if os.path.exists(file_actor) and os.path.exists(file_critic):
            self.actor.load_state_dict(torch.load(file_actor))
            self.critic.load_state_dict(torch.load(file_critic))
        # else:
            # self.logger.write_to_log(".pth doesn't exist. Use default parameters")


class MADDPG_PE(object):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__(self, state_dim, action_dim, max_velocity, agent_n, logger):
         # 存在于 GPU 的神经网络
        self.actor = ActorPedestrian(state_dim, action_dim, max_velocity).to(self.device)    # origin_network
        self.actor_target = ActorPedestrian(state_dim, action_dim, max_velocity).to(self.device)    # target_network
        self.actor_target.load_state_dict(self.actor.state_dict())  # initiate actor_target with actor's parameters
        # pytorch 中的 tensor 默认requires_grad 属性为false，即不参与梯度传播运算，特别地，opimizer中模型参数是会参与梯度优化的
        self.actor_optimizer = optim.Adam(self.actor.parameters(), pdata.LEARNING_RATE) # 以pdata.LEARNING_RATE指定学习率优化actor中的参数 

        self.critic = CriticCentralPE(agent_n).to(self.device)
        self.critic_target = CriticCentralPE(agent_n).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), pdata.LEARNING_RATE)
        # self.replay_buffer 取消：每个Agent不再有独立的经验池
        self.writer = SummaryWriter(pdata.DIRECTORY+'runs')
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.logger = logger


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # numpy.ndarray.flatten(): 返回一个 ndarray对象的copy，并且将该ndarray压缩成一维数组
        action = self.actor(state).cpu().data.numpy().flatten()
        return action       # 未归一化的

    def select_target_actions(self, states):
        # state 的输入必须是归一化的[[],[]] np.ndarray
        united_state = torch.FloatTensor(states).to(self.device)
        action = self.actor_target(united_state).cpu().data.numpy()
        return action   # 未归一化的

    def select_current_actions(self, states):
        united_state = torch.FloatTensor(states).to(self.device)
        action = self.actor(united_state).cpu().data.numpy()
        return action   # 未归一化的


    # update the parameters in actor network and critic network
    def update(self, central_replay_buffer_ma, agent_list, agent_idx):
        critic_loss_list = []
        actor_performance_list = []
        # Sample replay buffer:  [(united_normalized_states, united_normalized_next_states, united__normalized_action, [reward_1, ...], done), (...)]
        x, y, u, r, d = central_replay_buffer_ma.sample(pdata.BATCH_SIZE)     # 随机获取 batch_size 个五元组样本(sample random minibatch)

        next_actions = [agent_list[i].select_target_actions(y[:,i*pdata.STATE_DIMENSION : i*pdata.STATE_DIMENSION + pdata.STATE_DIMENSION])
                         for i in range(len(agent_list))]    
        next_actions = np.concatenate(next_actions, axis=1)
        
        united_next_action_batch = torch.FloatTensor(next_actions).to(self.device)
        united_state_batch = torch.FloatTensor(x).to(self.device)    
        united_action_batch = torch.FloatTensor(u).to(self.device)   
        united_next_state_batch = torch.FloatTensor(y).to(self.device)   
        done = torch.FloatTensor(d).to(self.device) 
        reward_batch = torch.FloatTensor(r[:,agent_idx])  # torch.Size([64])
        reward_batch = reward_batch[:, None].to(self.device)    # torch.Size([64,1])   
        target_Q = self.critic_target(united_next_state_batch, united_next_action_batch)
        target_Q = reward_batch + ((1 - done) * pdata.GAMMA * target_Q).detach()  # batch_size个维度
        current_Q = self.critic(united_state_batch, united_action_batch) 

        critic_loss = F.mse_loss(current_Q, target_Q) 
        self.writer.add_scalar('critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
        # self.logger.write_to_log('critic_loss:{loss}'.format(loss=critic_loss))
        # self.logger.add_to_critic_buffer(critic_loss.item())
        critic_loss_list.append(critic_loss.item())
        self.critic_optimizer.zero_grad()   # zeros the gradient buffer
        critic_loss.backward()              # back propagation on a dynamic graph
        self.critic_optimizer.step()

        current_actions = [agent_list[i].select_current_actions(x[:,i*pdata.STATE_DIMENSION : i*pdata.STATE_DIMENSION + pdata.STATE_DIMENSION])
                             for i in range(len(agent_list))]
        current_actions_batch = torch.FloatTensor(np.concatenate(current_actions, axis=1)).to(self.device)
        actor_loss = -self.critic(united_state_batch, current_actions_batch).mean()  
        self.writer.add_scalar('actor_performance', actor_loss, global_step=self.num_actor_update_iteration)
        # self.logger.write_to_log('actor_loss:{loss}'.format(loss=actor_loss))
        # self.logger.add_to_actor_buffer(actor_loss.item())
        actor_performance_list.append(actor_loss.item())

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

        actor_performance = np.mean(np.array(actor_performance_list)).item()
        self.logger.add_to_actor_buffer(actor_performance)
        critic_loss = np.mean(critic_loss_list).item()
        self.logger.add_to_critic_buffer(critic_loss)


    def save(self, mark_str):
        torch.save(self.actor.state_dict(), pdata.DIRECTORY + mark_str + '_actor_ma_pe.pth')
        torch.save(self.critic.state_dict(), pdata.DIRECTORY +  mark_str + '_critic_ma_pe.pth')
        

    def load(self, mark_str):
        file_actor = pdata.DIRECTORY +  mark_str + '_actor_ma_pe.pth'
        file_critic = pdata.DIRECTORY +  mark_str + '_critic_ma_pe.pth'
        if os.path.exists(file_actor) and os.path.exists(file_critic):
            self.actor.load_state_dict(torch.load(file_actor))
            self.critic.load_state_dict(torch.load(file_critic))
        # else:
            # self.logger.write_to_log(".pth doesn't exist. Use default parameters")