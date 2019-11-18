# implement by p0werHu
# time 11/18/2019


class Config(object):

    def __init__(self):
        # define some configures here
        self.x_dim = 28
        self.h_dim = 100
        self.z_dim = 16
        self.train_epoch = 100
        self.save_every = 10
        self.batch_size = 512
        self.device_ids = [0, 1]
        self.checkpoint_path = '../checkpoint/Epoch_61.pth'
        self.restore = True
