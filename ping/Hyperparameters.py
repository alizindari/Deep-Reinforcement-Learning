


class Hyperparameters():
    def __init__(self):
        self.IMAGESIZE = [30,30]
        self.MEAN_REWARD_BOUND = 19.0           
        self.CHANNEL_NUM = 4
        self.ACTION_SPACE = 6
        self.GAMMA = 0.99                   
        self.BATCH_SIZE = 32 
        self.REPLAY_SIZE = 10000 
        self.REPLAY_START_SIZE = 10000           
        self.LEARING_RATE = 1e-4 *1    
        self.SYNC_TARGET_FRAMES = 1000      
        ######################################      
        self.EPS_INITIAL = 1.0
        self.FIXED_EPSILON = 0.8   
        self.THRESH_START_DECAY = 20 
        self.EPS_DECAY = 0.99
        self.EPS_MIN = 0.02
        ######################################
        self.ITER_NUM = 80
        self.EPISODE_NUM = 500
        self.ITER_SUCCESS_FILL = 100
