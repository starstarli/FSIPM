import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_dataset
from data_loader import load_data_kflod
from loss_function import Triplet_loss
from net import FSIPM
from test import test
from valid import valid

import numpy as np
import random


import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, criterions, optimizer, train_loader, valid_loader, fold, epochs, num_classes,topk):
    train_size = len(train_loader)
    best_acc = 0
    best_model = None

    epochs_sum_loss = []
    epochs_ce_loss = []
    epochs_triplt_loss = []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        epoch_loss = 0

        CE_loss = 0
        triplet_loss = 0

        # train
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            label = labels.view(-1).cuda()
            embedding, x = model(inputs)
            predict = model.frozen_forward(inputs)

            loss1 = criterions[0](predict, label)
            loss2 = criterions[1](embedding, label)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            CE_loss += loss1.item()
            triplet_loss += loss2.item()

        # validation
        valid_acc, valid_sen, valid_spe, valid_auc = valid(train_loader, valid_loader, model, num_classes,topk=topk)

        if valid_acc >= best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, "best_epoch_fold{}.pkl".format(fold))
        epoch_loss = epoch_loss / train_size

        end = time.time() - start
        print("< F{} {:.0f}% {}/{} {:.3f}s >".format(fold, (epoch + 1) / epochs * 100, epoch + 1, epochs, end), end="")
        print('train_loss =', '{:.5f}'.format(epoch_loss), end="")
        print('valid_acc =', '{:.4f}'.format(valid_acc * 100))

        epochs_sum_loss.append(epoch_loss)
        epochs_ce_loss.append(CE_loss)
        epochs_triplt_loss.append(triplet_loss)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
   
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)

if __name__ == '__main__':
    #  Data paths for the all-site or sub-sites
    file_path = '/data'
    file_name = 'ALL&IS_aal3m161.mat'
    # Parameters
    timepoints, rois ,p = 90, 161 , 4
    dim, depth, heads = 256, 1, 1
    dropout = 0.5
    batch_size = 128
    epochs = 50
    num_classes = 2
    topk = 5

    seed = 3407
    same_seeds(seed)#3407
    pick = [0, 1]

    # get data
    data, pheno,label = load_dataset(file_path, file_name, pick=pick)
    valid_ratio = 0.2
    alpha, beta = 0.5, 0.5

    # k-fold validation
    predict_acc, predict_auc, predict_sen, predict_spe = [], [], [], []


    K = 5
    for ki in range(K):

        train_loader, valid_loader, test_loader, eval_loader = load_data_kflod(
            data, pheno, label,seed, batch_size=batch_size, num_workers=0,
            ki=ki, fold=K, valid_ratio=valid_ratio)
        model = FSIPM(rois,timepoints,p,num_classes,dropout,w=4).cuda()
        model.apply(weight_init)

        criterion1 = nn.CrossEntropyLoss().cuda()
        criterion2 = Triplet_loss(margin=0.8, loss_weight=alpha).cuda()
        criterions = [criterion1, criterion2]

        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)

        train(model, criterions, optimizer, train_loader, valid_loader, ki + 1, num_classes=num_classes, epochs=epochs,topk = topk)
        test_model = torch.load("best_epoch_fold{}.pkl".format(ki + 1))
        ACC, SEN, SPE, AUC = test(test_loader, eval_loader, test_model, num_classes,ki,topk = topk)
        predict_acc.append(ACC)
        predict_auc.append(AUC)
        predict_sen.append(SEN)
        predict_spe.append(SPE)
        print('test_acc ={:.4f},auc_acc ={:.4f},sen_acc ={:.4f},spe_acc ={:.4f}'.format(acc * 100,auc * 100,SEN * 100,SPE * 100))