import random, copy
import numpy as np
import public_data as pdata


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

    # buffer 中的 action 是未归一化的
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
            X_HER, Y_HER, U_HER, R_HER, D_HER = self.storage[(i+1) % self.max_size] 
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


class FilterReplayBuffer():
    def __init__(self, max_size=pdata.CAPACITY):
        self.storage = []
        self.max_size = max_size 
        self.ptr = 0
        self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size= pdata.HER_K)  # 在 0 到 len(self.storage) 之间产生 size 个数
        # state, next_state, action, reward, done
        x, y, u, r, d = [], [], [], [], []

        # 这里的样本是完全随机的batch_size个
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            X_HER, Y_HER, U_HER, R_HER, D_HER = self.storage[(i+1) % self.max_size] 
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

    # 增加筛选机制 —— 最长递增子序列
    def push(self, data_seq):
        # data_seq : [[next_state, state, action, reward, done]...]
        longest_subseq = []
        for i in range(0, len(data_seq)):
            tmp_seq = []    # 保存索引号
            tmp_seq.append(i)
            for j in range(i+1, len(data_seq)):
                if data_seq[j][3] > data_seq[len(tmp_seq)-1][3]:
                    tmp_seq.append(j)
            if len(longest_subseq) < len(tmp_seq):
                longest_subseq = copy.deepcopy(tmp_seq)

        for i in range(0, len(longest_subseq)):
            self._push((data_seq[i][0], data_seq[i][1], data_seq[i][2], data_seq[i][3], data_seq[i][4]))

    # 真正放入经验池的机制
    def _push(self, data_tuple):
        if len(self.storage) == self.max_size:
            # 经验回放池的超限更新策略
            self.storage[int(self.ptr)] = data_tuple
            self.ptr = (self.ptr + 1) % self.max_size       #循环数组更新
        else:
            self.storage.append(data_tuple)


class CentralReplayBufferMA():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=pdata.CAPACITY):
        # buffer data: [(united_normalized_states, united_normalized_next_states, united__normalized_action, [reward_1, ...], done), (...)]
        # united_normalized_states: numpy.ndarray
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    # buffer 中的 action 是未归一化的
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
            x.append(X) 
            y.append(Y)
            u.append(U)
            r.append(np.array(R, copy=False))   # R is a list
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r), np.array(d).reshape(-1, 1)