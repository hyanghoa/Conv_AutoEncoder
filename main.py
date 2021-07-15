import os
import random

import numpy as np
import torch

from model import AutoEncoder
from dataset import Dataset
from train import train, val
from log import CreateLog


TRAIN_FILE_LIST = "./dataset_list/train.lst"
VAL_FILE_LIST = "./dataset_list/val.lst"
TOTAL_EPOCH = 300
BATCH_SIZE = 8
LR = 0.001
WEIGHT_DECAY = 0.9
DEVICE_IDS = [0, 1]
MODEL_NAME = "test"
LOG_DIR = f"./outputs/{MODEL_NAME}/logs"
RANDOMNESS = False


# random control
if RANDOMNESS:
    random_seed = 12361

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# make dir
os.makedirs(LOG_DIR, exist_ok=True)

# create log
log = CreateLog(LOG_DIR)

# create model
model = AutoEncoder()

if torch.cuda.is_available():
    model.cuda()

model.init_weights()
model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)

# dataset
train_dataset = Dataset(file_list=TRAIN_FILE_LIST)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

val_dataset = Dataset(file_list=VAL_FILE_LIST)
val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=10, 
                                                                 T_mult=1,
                                                                 eta_min=0.0000001)

for current_peoch in range(1, TOTAL_EPOCH):
    train(current_peoch, TOTAL_EPOCH, train_loader, optimizer, scheduler, model, log)
    val(val_loader, model, log)

    torch.save(model.module.state_dict(),
                os.path.join(OUTPUT_DIR, f"{current_peoch}.pth"))
    log.write(f"{current_peoch} epoch model saved!")
