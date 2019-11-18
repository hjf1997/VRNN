# implemented by p0werHu
# 11/15/2019

import  torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from utils import Progbar
from model import VRNN
from loss_function import loss as Loss
from config import Config

def load_dataset(batch_size):
    train_dataset = datasets.MNIST(root='../data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='../data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader

def train(conf):

    train_loader, test_loader = load_dataset(512)
    net = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(112858)
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    if conf.restore == True:
        net.load_state_dict(torch.load(conf.checkpoint_path, map_location='cuda:0'))
        print('Restore model from ' + conf.checkpoint_path)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for ep in range(1, conf.train_epoch + 1):
        prog = Progbar(target=117)
        print("At epoch:{}".format(str(ep)))
        for i, (data, target) in enumerate(train_loader):
            data = data.squeeze(1)
            data = (data / 255).to(device)
            package = net(data)
            loss = Loss(package, data)
            net.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            prog.update(i, exact=[("Training Loss", loss.item())])

        with torch.no_grad():
            x_decoded = net.module.sampling(conf.x_dim, device)
            x_decoded = x_decoded.cpu().numpy()
            digit = x_decoded.reshape(conf.x_dim, conf.x_dim)
            plt.imshow(digit, cmap='Greys_r')
            plt.pause(1e-6)

        if ep % conf.save_every == 0:
            torch.save(net.state_dict(), '../checkpoint/Epoch_' + str(ep + 1) + '.pth')

def generating(conf):

    net = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=conf.device_ids)
    net.load_state_dict(torch.load(conf.checkpoint_path, map_location='cuda:0'))
    print('Restore model from ' + conf.checkpoint_path)

    with torch.no_grad():
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        for i in range(n):
            for j in range(n):
                x_decoded = net.module.sampling(digit_size, device)
                x_decoded = x_decoded.cpu().numpy()
                digit = x_decoded.reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()



if __name__ == '__main__':

    conf = Config()
    train(conf)
    generating(conf)





