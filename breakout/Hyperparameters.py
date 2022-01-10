


class Hyperparameters():
    def __init__(self):
        self.IMAGESIZE = [84,84]
        self.CHANNEL_NUM = 4
        self.ACTION_SPACE = 4
        self.GAMMA = 0.99                   
        self.BATCH_SIZE = 32 
        self.REPLAY_SIZE = 1000 
        self.SUCCESS_REPLAY_SIZE = 400 
        self.REPLAY_START_SIZE = 200           
        self.LEARING_RATE = 1e-4 
        self.SYNC_TARGET_FRAMES = 1000      
        ######################################      
        self.EPS_INITIAL = 1.0
        self.FIXED_EPSILON = 0.8   
        self.THRESH_START_DECAY = 10 
        self.EPS_DECAY = 0.99
        self.EPS_MIN = 0.02
        self.SUCCESS_SAMPLE_PROB = 0.2
        ######################################
        self.ITER_NUM = 400
        self.EPISODE_NUM = 400
        self.ITER_SUCCESS_FILL = 600
