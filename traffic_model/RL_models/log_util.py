import os,sys,time
import public_data as pdata


''' 文件的保存路径 '''
DIRECTORY = pdata.DIRECTORY

class logWriter():
    ''' 文件生成的时间 '''
    def __init__(self, mark_str):
        self.START_TIME = time.strftime("%Y-%m-%d_%H-%M", time.localtime(time.time()))
        self.EXPERIMENT_LOG = open(DIRECTORY + self.START_TIME + '_' + mark_str +'.txt', 'a+')

    def __del__(self):
        self.record_end_time()
        self.close_file()

    def record_start_time(self):
        self.write_to_log('--------------START TIME: '+ self.START_TIME + '---------------------')

    def record_end_time(self):
        self.write_to_log('END TIME: {end}'.format(end=time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime(time.time()))))

    def write_to_log(self, content_str):
        print(content_str, file = self.EXPERIMENT_LOG)

    def close_file(self):
        self.EXPERIMENT_LOG.close()

    def record_hard_params(self):
        self.write_to_log('===============  initialization parameters  ===============')
        self.write_to_log('experiment: episodes = {e}, len_of_trajectory = {lt}, gamma = {g}, learning_rate = {lr}, exploration_rate = {er},\n buffer_size = {bs}, batch_size = {bs1}'.format(
            e = pdata.MAX_EPISODE, lt = pdata.MAX_LENGTH_OF_TRAJECTORY, g = pdata.GAMMA, lr = pdata.LEARNING_RATE, er = pdata.EXPLORATION_NOISE, bs = pdata.CAPACITY, bs1 = pdata.BATCH_SIZE
        ))
        self.write_to_log('environment limits : max_velocity = {mv} m/frame, fps = {fps} frames/s'.format(
            mv = pdata.VELOCITY_LIMIT, fps = pdata.FPS
        ))
        self.write_to_log('===========================================================')