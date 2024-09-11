import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_dataset
from data_loader import load_data_is
from loss_function import MyTriplet_loss
from net import FSIPM
from test import test
from valid import valid
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, criterions, optimizer, train_loader, epochs, num_classes,topk):
    train_size = len(train_loader)
    best_model = None
    best_acc = 0

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
            torch.save(best_model, "finnal_epoch_model.pkl")
        epoch_loss = epoch_loss / train_size

        end = time.time() - start
        print("< {:.0f}% {}/{} {:.3f}s >".format((epoch + 1) / epochs * 100, epoch + 1, epochs, end), end="")
        print('train_loss =', '{:.5f}'.format(epoch_loss))

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
    ADHD_train = '/data/ALL_aal3m161.mat'
    ADHD_test = '/data/ALL_ISaal3m161.mat'

    # Parameters
    timepoints, rois,p = 172, 161, 4
    dim, depth, heads = 256, 1, 1
    dropout = 0.5
    batch_size = 128
    epochs = 50
    num_classes = 2
    seed = 3407
    topk = 5
    same_seeds(seed)#3407
    pick = [0, 1]

    # get data
    data_train, pheno_train,label_train = load_dataset(ADHD_train, timepoints = timepoints,pick=pick)
    data_test, pheno_test,label_test = load_dataset(ADHD_test,timepoints = timepoints,pick=pick)

    valid_ratio = 0.2
    alpha, beta = 0.5, 0.5

    predict_acc, predict_auc, predict_sen, predict_spe = [], [], [], []

    train_loader,valid_loader,eval_loader = load_data_is(data_train, pheno_train,label_train, batch_size=batch_size, num_workers=0,valid_ratio=valid_ratio)
    test_loader = load_data_is(data_test,pheno_test, label_test, batch_size=batch_size, num_workers=0,valid_ratio=-1)

    model = FSIPM(rois,timepoints=90,p=p,num_classes=num_classes,dropout=dropout,w=4).cuda()
    model.apply(weight_init)

    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = MyTriplet_loss(margin=0.8, loss_weight=alpha).cuda()
    criterions = [criterion1, criterion2]

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)

    train(model, criterions, optimizer, train_loader,num_classes=num_classes, epochs=epochs,topk = topk)
    test_model = torch.load("finnal_epoch_model.pkl")
    acc, SEN, SPE, auc = test(test_loader,train_loader, test_model, num_classes,topk = topk)

    predict_acc.append(acc)
    predict_auc.append(auc)
    predict_sen.append(SEN)
    predict_spe.append(SPE)
    print('test_acc ={:.4f},auc_acc ={:.4f},sen_acc ={:.4f},spe_acc ={:.4f}'.format(acc * 100,auc * 100,SEN * 100,SPE * 100))