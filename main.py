import os
import random

import numpy as np
import torch
import segmentation_models_pytorch as smp

from models import efficientnetv2, efficientnetv2_test
from runs.train import train, val
from utils.dataset import Dataset
from utils.log import CreateLog


TRAIN_FILE_LIST = "./dataset_list/192_train.lst"
VAL_FILE_LIST = "./dataset_list/192_val.lst"
TOTAL_EPOCH = 100
BATCH_SIZE = 96
LR = 1e-5
WEIGHT_DECAY = 1e-6
DEVICE_IDS = [0, 1]
MODEL_NAME = "effcientnet_unet_resize"
OUTPUT_DIR = f"./outputs/{MODEL_NAME}"
LOG_DIR = f"./outputs/{MODEL_NAME}/logs"
RANDOMNESS = False
PRETRAINED = False
PRETRAINED_MODEL = "/home/djlee/autoencoder/outputs/effcientnet_unet_resize/final.pth"

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
os.makedirs(os.path.join(OUTPUT_DIR, "training_images"), exist_ok=True)

# create log
log = CreateLog(LOG_DIR)

# create model
model = efficientnetv2_test.EfficientNetV2Unet("tf_efficientnetv2_xl_in21ft1k")
# model = smp.UnetPlusPlus(
#             encoder_name="tu-tf_efficientnetv2_xl_in21ft1k",
#             decoder_channels=[640, 256, 96, 64, 32],
#             decoder_attention_type="scse",
#             in_channels=3,
#             classes=3,
#         )
# print(model)

# load pretrained model
if PRETRAINED:
    model.load_state_dict(torch.load(PRETRAINED_MODEL))

if torch.cuda.is_available():
    model.cuda()

model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)

# optimizer
scaler = torch.cuda.amp.GradScaler()
loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=200)

log.write(
    f"""
    TRAIN_FILE_LIST: {TRAIN_FILE_LIST},
    VAL_FILE_LIST: {VAL_FILE_LIST},
    TOTAL_EPOCH: {TOTAL_EPOCH},
    BATCH_SIZE: {BATCH_SIZE},
    LR: {LR},
    WEIGHT_DECAY: {WEIGHT_DECAY},
    DEVICE_IDS: {DEVICE_IDS},
    MODEL_NAME: {MODEL_NAME},
    OUTPUT_DIR: {OUTPUT_DIR},
    LOG_DIR: {LOG_DIR},
    RANDOMNESS: {RANDOMNESS},
    PRETRAINED: {PRETRAINED},
    PRETRAINED_MODEL: {PRETRAINED_MODEL}
    """
)
best_psnr = 0
for current_peoch in range(0, TOTAL_EPOCH):
    if current_peoch < 20:
        TRAIN_FILE_LIST = "./dataset_list/192_train.lst"
        VAL_FILE_LIST = "./dataset_list/192_val.lst"
        BATCH_SIZE = 128
    elif 20 <= current_peoch < 50:
        TRAIN_FILE_LIST = "./dataset_list/384_train.lst"
        VAL_FILE_LIST = "./dataset_list/384_val.lst"
        BATCH_SIZE = 32
    elif 50 <= current_peoch:
        TRAIN_FILE_LIST = "./dataset_list/512_train.lst"
        VAL_FILE_LIST = "./dataset_list/512_val.lst"
        BATCH_SIZE = 16

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
            val_dataset,
            batch_size=BATCH_SIZE//4,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True)

    train(current_peoch, TOTAL_EPOCH, train_loader, optimizer, scheduler, model, loss, l1_loss, scaler, log)
    avg_psnr = val(val_loader, model, loss, log)

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.module.state_dict(), os.path.join(OUTPUT_DIR, 'best.pth'))
        log.write(f"=> saving best checkpoint to {OUTPUT_DIR}/best.pth")

    torch.save({
        'epoch': current_peoch+1,
        'best_psnr': best_psnr,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(f"{OUTPUT_DIR}", 'checkpoint.pth.tar'))
    log.write(f"=> saving checkpoint to {OUTPUT_DIR}/checkpoint.pth.tar")


    torch.save(model.module.state_dict(), os.path.join(f"{OUTPUT_DIR}", f"final.pth"))
    log.write(f"{current_peoch} epoch model saved!")
