import os
import scipy.io as scio
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

def statistics(y_true, pre):
    acc, auc, sen, spe = 0.0, 0.0, 0.0, 0.0
    try:
        ACC = accuracy_score(y_true, pre)
        AUC = roc_auc_score(y_true, pre)
        TP = torch.sum(y_true & pre)
        TN = len(y_true) - torch.sum(y_true | pre)
        true_sum = torch.sum(y_true)
        neg_sum = len(y_true) - true_sum
        SEN = TP / true_sum
        SPE = TN / neg_sum

        acc += ACC
        sen += SEN.cpu().numpy()
        spe += SPE.cpu().numpy()
        auc += AUC

    except ValueError as ve:
        print(ve)
        pass

    return acc, sen, spe, auc


def pick_data(data, pick):
    picked_data = []
    for i in range(len(data)):
        if data[i][-1] in pick:
            picked_data.append(data[i])
    return np.array(picked_data)


def change_label(labels):
    if 0 not in labels:
        for i, label in enumerate(labels):
            labels[i] -= 1
    else:
        for i, label in enumerate(labels):
            if labels[i] != 0:
                labels[i] = 1
    return labels



def fmri_processing(fmri):
    b,n,m= fmri.shape
    # fmri = np.exp(-(fmri**2/deta))
    # fmri = fishers_r_to_z(fmri)
    return fmri

def smri_processing(smri):
    b, n = smri.shape
    rois_colum = smri.reshape(b,-1,n)
    rois_row = smri.reshape(b,n,-1)
    smri = (rois_colum - rois_row)
    smri = 1/(smri**2+1)
    return smri

def Preprocessing(fmri,smri,label):
    fmri = fmri_processing(fmri)
    smri = smri_processing(smri)
    label = np.array([1 if flag >= 2 else flag for flag in label.reshape(-1)])
    return fmri,smri,label

def load_dataset(file_path, file_name, pick=[1, 2]):
    adhd = scio.loadmat(os.path.join(file_path, file_name))

    fmri = adhd['time_series'].transpose(0,2,1)
    smri = adhd['vgm']
    pheno = np.concatenate((adhd['Gender'],adhd['Age'],adhd['Handedness'],adhd['Full4_IQ']),axis = 0).transpose(1,0)
    label = adhd['label']
    
    fmri,smri,label = Preprocessing(fmri,smri,label)

    b, h, w = fmri.shape
    _, n, m = smri.shape
    _,p = pheno.shape

    fmri = fmri.reshape(b, -1)
    smri = smri.reshape(b, -1)
    pheno = pheno.reshape(b,-1)
    label = label.reshape(b,-1)

    data = np.concatenate((fmri, smri, pheno ,label), axis=1)

    data = pick_data(data, pick)  
    fmri = data[:, :h * w]
    smri = data[:, h * w: -(p+1)]
    pheno = data[:, -(p+1):-1]

    fmri = fmri.reshape(-1, h, w)
    smri = smri.reshape(-1, n, m)
    pheno = pheno.reshape(-1,p)
    label = data[:, -1]

    return np.concatenate((fmri, smri), axis=2), pheno,label


def my_split_dataset(data, pheno,label, seed,ki, K, valid_ratio):
    num_folds = K
    shuffle = True
    b, h, w = data.shape
    data = data.reshape(b, -1)
    pheno = pheno.reshape(b, -1)
    data = np.concatenate((data, pheno, label.reshape(b, -1)), axis=1)

    splitter = StratifiedKFold(num_folds, shuffle=shuffle, random_state=seed)
    splits = list(splitter.split(range(len(data)), y=label))
    valid_index = random.sample(list(splits[ki][0]),int(len(splits[ki][0])*valid_ratio))
    train_index = list(set(splits[ki][0]) - set(valid_index))

    trainset = data[train_index]
    validset = data[valid_index]
    testset = data[splits[ki][1]]
    
    np.random.shuffle(trainset)
    np.random.shuffle(testset)
    np.random.shuffle(validset)
    return trainset, validset, testset