import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolo3 import YoloBody
from nets.yolo_training import Generator, YOLOLoss
from utils.config import Config
from utils.train_details import train_details
from utils.yolo_dataset import YoloDataset, yolo_dataset_collate

min_loss = 999999999.99


def fit_ont_epoch(net, yolo_losses, epoch, train_batch_num, val_batch_num, train_gen, val_gen, Epoch, cuda):
    global min_loss

    total_loss = 0
    val_loss = 0

    print('\nEpoch:{}/{}'.format(epoch + 1, Epoch))
    print('# Start Training')

    net.train()
    for iteration, batch in enumerate(train_gen):
        if iteration >= train_batch_num:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = net(images)

        losses = []
        num_pos_all = 0
        # ----------------------#
        #   计算损失
        # ----------------------#
        for i in range(3):
            loss_item, num_pos = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item)
            num_pos_all += num_pos

        loss = sum(losses) / num_pos
        # ----------------------#
        #   反向传播
        # ----------------------#
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print('# Start Validation')
    net.eval()
    for iteration, batch in enumerate(val_gen):
        if iteration >= val_batch_num:
            break
        images_val, targets_val = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            else:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            num_pos_all = 0
            for i in range(3):
                loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item)
                num_pos_all += num_pos
            loss = sum(losses) / num_pos
            val_loss += loss.item()

    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (train_batch_num + 1), val_loss / (val_batch_num + 1)))

    # 每一个epoch都存储
    # print('Saving state, iter:', str(epoch + 1))
    # torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    #     (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    if val_loss < min_loss:
        min_loss = val_loss
        print('** saving best model **')
        dst_path = 'model/' + train_details["model_name"] + '.pth'
        torch.save(model.state_dict(), dst_path)

    return val_loss / (val_batch_num + 1)


if __name__ == "__main__":
    Cuda = True
    # --------------------------------1. 模型载入--------------------------------------------
    # 对损失进行归一化
    normalize = True

    model = YoloBody(Config)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on cuda')
    else:
        device = torch.device('cpu')
        print('Running on cpu')

    net = model.train()
    # GPU并行
    net = torch.nn.DataParallel(model)
    net = net.to(device)

    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的。
    cudnn.benchmark = True

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]), Config["yolo"]["classes"],
                                    (Config["img_w"], Config["img_h"]), Cuda, normalize))

    # --------------------------------2. 数据准备--------------------------------------------
    # 获得图片路径和标签
    dataset_path = './data/dataset.txt'

    # 划分数据集
    val_split = train_details["validation_rate"]
    with open(dataset_path) as f:
        lines = f.readlines()
    np.random.seed(0)
    np.random.shuffle(lines)

    # 划分成两部分
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    val_lines = lines[num_train:]
    np.save('./data/val_lines.npy', val_lines)
    train_lines = lines[:num_train]
    np.save('./data/train_lines.npy', train_lines)

    # 参数
    batch_size = train_details["batch_size"]

    # Dataset
    train_dataset = YoloDataset(train_lines, (Config["img_h"], Config["img_w"]), is_train=True)
    val_dataset = YoloDataset(val_lines, (Config["img_h"], Config["img_w"]), is_train=False)
    # DataLoader
    train_gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                           drop_last=True, collate_fn=yolo_dataset_collate)
    val_gen = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    # --------------------------------3. 模型训练--------------------------------------------
    lr = train_details["learning_rate"]
    optimizer = optim.Adam(net.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train_batch_num = num_train // batch_size
    val_batch_num = num_val // batch_size

    val_losses = []

    Epoch = train_details["epoch"]
    for epoch in range(Epoch):
        cur_loss = fit_ont_epoch(net, yolo_losses, epoch, train_batch_num, val_batch_num, train_gen, val_gen, Epoch,
                                 Cuda)
        val_losses.append(cur_loss)

        lr_scheduler.step()

        np.savetxt('val_losses.txt', np.array(val_losses))
