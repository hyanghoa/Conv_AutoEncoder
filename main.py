import torch
from model import AutoEncoder
from dataset import Dataset
from train import train

TRAIN_FILE_LIST = "./dataset_list/imagenet_train.lst"
VAL_FILE_LIST = "./dataset_list/imagenet_val.lst"
TOTAL_EPOCH = 300
BATCH_SIZE = 24
LR = 0.00001
DEVICE_IDS = [0, 1]

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
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

# optimizer
params_dict = dict(model.named_parameters())
params = [{'params': list(params_dict.values()), 'lr': LR}]
optimizer = torch.optim.Adam(params, lr=LR)

for current_peoch in range(1, TOTAL_EPOCH):
    train(current_peoch, TOTAL_EPOCH, train_loader, optimizer, model)
