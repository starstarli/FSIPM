import numpy as np
import torch
import torch.utils.data as Data
from utils import change_label,my_split_dataset
import random


class GetKfoldLoader(Data.Dataset):
    def __init__(self, data):
        super(GetKfoldLoader, self).__init__()
        self.data = data[:, : -1]
        self.label = data[:, -1].astype(np.int)
        self.label = change_label(self.label)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.int64)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.label[item]

def data_augment(datas, datashape ,phenoshape):
    _,rois,n = datashape
    _,p = phenoshape
    data = datas[:, : -p -1].reshape(-1,rois,n)
    label = datas[:, -1]
    pheno = datas[:, -p:]
    
    win,crops =90, 10
    
    augment_data = []
    augment_label = []
    augment_pheno = []

    range_list = range(90 + 1, n - rois)
    random_index = random.sample(range_list, crops)
    for i in range(len(data)):
        fmri = data[i, : , :-rois]
        smri = data[i, : , -rois:]
        for r in random_index:
            augment_data.append(np.concatenate((fmri[:,r - win:r],smri),axis = 1))
            augment_pheno.append(pheno[i])
            augment_label.append(label[i])
    b = datas.shape[0] * crops
    augment_data = np.array(augment_data).reshape(b,-1)
    augment_pheno = np.array(augment_pheno).reshape(b,-1)
    augment_label = np.array(augment_label).reshape(b,-1)
    return np.concatenate((augment_data,augment_pheno,augment_label),axis=1)

def load_data_kflod(data, pheno , label,seed, batch_size, num_workers, ki, fold, valid_ratio=0.15):
    trainset, validset, testset = my_split_dataset(data, pheno,label,seed, ki, fold, valid_ratio)
    trainset, validset, testset = data_augment(trainset,datashape=data.shape,phenoshape=pheno.shape),data_augment(validset,datashape=data.shape,phenoshape=pheno.shape),data_augment(testset,datashape=data.shape,phenoshape=pheno.shape)
    
    evalset = np.concatenate((trainset, validset), axis=0)
    train_loader = GetKfoldLoader(data=trainset)
    valid_loader = GetKfoldLoader(data=validset)
    test_loader = GetKfoldLoader(data=testset)
    eval_loader = GetKfoldLoader(data=evalset)

    train_dataloader = Data.DataLoader(
        dataset=train_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )
    valid_dataloader = Data.DataLoader(
        dataset=valid_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    test_dataloader = Data.DataLoader(
        dataset=test_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    eval_dataloader = Data.DataLoader(
        dataset=eval_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, valid_dataloader, test_dataloader, eval_dataloader

def load_data_is(data, pheno, label, batch_size, num_workers,valid_ratio):
    b, h, w = data.shape
    data = data.reshape(b, -1)
    dataset = np.concatenate((data, pheno,label.reshape(b, -1)), axis=1)

    testset = data_augment(dataset,datashape=[b,h,w],phenoshape=pheno.shape)
    test_loader = GetKfoldLoader(data = testset, datashape=[1,161,251])
    test_loader = Data.DataLoader(
        dataset=test_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    if valid_ratio < 0 :
        return test_loader
    print(dataset.shape,data.shape)
    valid_index = random.sample(range(0,b),int(b * valid_ratio))
    train_index = list(set(range(0,b))- set(valid_index))
    trainset = dataset[train_index]
    validset = dataset[valid_index]
    trainset, validset = data_augment(trainset,datashape=[b,h,w],phenoshape=pheno.shape),data_augment(validset,datashape=[b,h,w],phenoshape=pheno.shape),
    evalset = np.concatenate((trainset, validset), axis=0)

    np.random.shuffle(trainset)
    np.random.shuffle(validset)

    train_loader = GetKfoldLoader(data = trainset, datashape=[1,161,251])
    valid_loader = GetKfoldLoader(data = validset, datashape=[1,161,251])
    eval_loader = GetKfoldLoader(data=evalset, datashape=[1,161,251])
    
    train_loader = Data.DataLoader(
        dataset=train_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )
    valid_loader = Data.DataLoader(
        dataset=valid_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    eval_loader = Data.DataLoader(
        dataset=eval_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    

    return train_loader,valid_loader,eval_loader