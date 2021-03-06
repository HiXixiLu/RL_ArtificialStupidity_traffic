import time,copy
import threading
import public_data as pdata
import pandas
import numpy as np


''' 文件的保存路径 '''
DIRECTORY = pdata.DIRECTORY

class logWriter():
    ''' 文件生成的时间 '''
    def __init__(self, mark_str):
        self.START_TIME = time.strftime("%m-%d_%H-%M", time.localtime(time.time()))
        self._file_name = mark_str
        self._critic_loss = []
        self._actor_loss = []
        self._position_seq = []
        self._tmp_seq = []
        self.loss_counter = 0
        self.position_counter = 0
        # self.EXPERIMENT_LOG = open(DIRECTORY + self.START_TIME + '_' + mark_str +'.txt', 'w')
        txt_file = open(DIRECTORY + self._file_name +'_params.txt', 'a')
        txt_file.write('--------------START TIME: '+ self.START_TIME + '---------------------')
        txt_file.write('\n===============  initialization parameters  ===============\n')
        txt_file.write('\nexperiment: episodes = {e}, len_of_trajectory = {lt}, gamma = {g}, learning_rate = {lr}, exploration_rate = {er},\nbuffer_size = {bs}, batch_size = {bs1}'.format(
            e = pdata.MAX_EPISODE, lt = pdata.MAX_LENGTH_OF_TRAJECTORY, g = pdata.GAMMA, lr = pdata.LEARNING_RATE, er = pdata.EXPLORATION_NOISE, bs = pdata.CAPACITY, bs1 = pdata.BATCH_SIZE
        ))
        txt_file.write('\nenvironment limits : max_velocity = {mv} m/frame, fps = {fps} frames/s'.format(
            mv = pdata.VELOCITY_LIMIT, fps = pdata.FPS
        ))
        txt_file.write('\nbuffer_timer = ' + str(pdata.SLEEP_TIME) + ' s')
        txt_file.write('\n===========================================================')


    def clear_buffer(self):
        if len(self._critic_loss) > 0:
            # loss表格
            data_frame = pandas.DataFrame({'critic_loss':self._critic_loss, 'actor_loss':self._actor_loss})
            data_frame.to_csv(DIRECTORY + self._file_name + '_loss' + str(self.loss_counter) + '.csv', index=False)   # 有一个seq参数，如果不指定，则默认分割符为','
            self.loss_counter = self.loss_counter + 1
            self._critic_loss.clear()
            self._actor_loss.clear()

        if len(self._position_seq) > 0:
            # 路径点序列
            txt_file = open(DIRECTORY + self._file_name  + '_positions' + str(self.position_counter) + '.txt', 'w')
            txt_file.write('position sequences:')
            for i in range(0, len(self._position_seq)):
                ss = ''
                for j in range(0, len(self._position_seq[i])):
                    ss = ss + str(self._position_seq[i][j][0]) + ',' + str(self._position_seq[i][j][1]) + ' '
                ss = ss[:-1]
                txt_file.write('\n'+ ss)
            txt_file.close()
            self._position_seq.clear()
            self.position_counter = self.position_counter + 1

        global timer 
        timer = threading.Timer(pdata.SLEEP_TIME, self.clear_buffer)
        timer.start()


    def add_to_position_seq(self):
        if len(self._tmp_seq) > 0:
            self._position_seq.append(copy.deepcopy(self._tmp_seq))
        self._tmp_seq.clear()

    def record_position(self, pos):
        # pos: numpy.ndarray
        tmp = np.around(pos, decimals=2)
        self._tmp_seq.append(tmp)

    def add_to_critic_buffer(self, num):
        self._critic_loss.append(num)

    def add_to_actor_buffer(self, num):
        self._actor_loss.append(num)


    def record(self):
        # 参数文件
        txt_file = open(DIRECTORY + self._file_name +'_params.txt', 'a')
        txt_file.write('\n--------------END TIME: '+ time.strftime("%m-%d_%H-%M", time.localtime(time.time())) + '---------------------')
        txt_file.close()    

        # loss表格
        data_frame = pandas.DataFrame({'critic_loss':self._critic_loss, 'actor_loss':self._actor_loss})
        data_frame.to_csv(DIRECTORY + self._file_name + '_loss' + str(self.loss_counter) + '.csv', index=False)   # 有一个seq参数，如果不指定，则默认分割符为','
        self._critic_loss.clear()
        self._actor_loss.clear()
        
        # 路径点序列
        txt_file = open(DIRECTORY + self._file_name  + '_positions' + str(self.position_counter) + '.txt', 'w')
        txt_file.write('position sequences:')
        for i in range(0, len(self._position_seq)):
            ss = ''
            for j in range(0, len(self._position_seq[i])):
                ss = ss + str(self._position_seq[i][j][0]) + ',' + str(self._position_seq[i][j][1]) + ' '
            ss = ss[:-1]
            txt_file.write('\n'+ ss)
        txt_file.close()
        self._position_seq.clear()
        self.position_counter = self.position_counter + 1


    def record_pe(self):
        # 参数文件
        txt_file = open(DIRECTORY + self._file_name +'_params.txt', 'a')
        txt_file.write('\n--------------END TIME: '+ time.strftime("%m-%d_%H-%M", time.localtime(time.time())) + '---------------------')
        txt_file.close()

        # loss表格
        data_frame = pandas.DataFrame({'critic_loss':self._critic_loss, 'actor_loss':self._actor_loss})
        data_frame.to_csv(DIRECTORY + self._file_name + '_loss.csv', index=False)   # 有一个seq参数，如果不指定，则默认分割符为','

        # 路径点序列
        txt_file = open(DIRECTORY + self._file_name  + '_positions.txt', 'w')
        txt_file.write('position sequences:')
        for i in range(0, len(self._position_seq)):
            ss = ''
            for j in range(0, len(self._position_seq[i])):
                ss = ss + str(self._position_seq[i][j][0]) + ',' + str(self._position_seq[i][j][1]) + ' '
            ss = ss[:-1]
            txt_file.write('\n' + ss)
        txt_file.close()


    # def record_start_time(self):
    #     self.write_to_log('--------------START TIME: '+ self.START_TIME + '---------------------')

    # def record_end_time(self):
    #     self.write_to_log('END TIME: {end}'.format(end=time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime(time.time()))))

    # def write_to_log(self, content_str):
    #     self.EXPERIMENT_LOG.write(content_str)

    # def close_file(self):
    #     self.EXPERIMENT_LOG.close()

    # def record_hard_params(self):
    #     self.write_to_log('===============  initialization parameters  ===============')
    #     self.write_to_log('experiment: episodes = {e}, len_of_trajectory = {lt}, gamma = {g}, learning_rate = {lr}, exploration_rate = {er},\n buffer_size = {bs}, batch_size = {bs1}'.format(
    #         e = pdata.MAX_EPISODE, lt = pdata.MAX_LENGTH_OF_TRAJECTORY, g = pdata.GAMMA, lr = pdata.LEARNING_RATE, er = pdata.EXPLORATION_NOISE, bs = pdata.CAPACITY, bs1 = pdata.BATCH_SIZE
    #     ))
    #     self.write_to_log('environment limits : max_velocity = {mv} m/frame, fps = {fps} frames/s'.format(
    #         mv = pdata.VELOCITY_LIMIT, fps = pdata.FPS
    #     ))
    #     self.write_to_log('===========================================================')
        
    