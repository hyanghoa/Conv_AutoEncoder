import torch
import numpy as np
from model import AutoEncoder
from dataset import Dataset
from train import train

FILE_LIST = "./dataset_list/train.lst"

# create model
model = AutoEncoder()
model = torch.nn.DataParallel(model, device_ids=[0, 1])
if torch.cuda.is_available():
    model.cuda()


# dataset
train_dataset = Dataset(file_list=FILE_LIST)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

# optimizer
params_dict = dict(model.named_parameters())
params = [{'params': list(params_dict.values()), 'lr': 0.00001}]
optimizer = torch.optim.Adam(params, lr=0.00001)

total_epoch = 100
for current_peoch in range(1, total_epoch):
    train(current_peoch, total_epoch, train_loader, optimizer, model)