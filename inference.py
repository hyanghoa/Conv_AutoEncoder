import os

import torch
from models import efficientnetv2, efficientnetv2_test
from utils.dataset import Dataset
from runs.train import inference, slide_inference


TEST_FILE_LIST = "./dataset_list/test.lst"
BATCH_SIZE = 20
DEVICE_IDS = [0, 1]
MODEL_NAME = "effcientnet_unet_resize"
OUTPUT_DIR = f"./outputs/{MODEL_NAME}/inference"
LOARD_MODEL = "/home/djlee/autoencoder/outputs/effcientnet_unet_resize/final.pth"

# make dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# create model
model = efficientnetv2_test.EfficientNetV2Unet("tf_efficientnetv2_xl_in21ft1k")

# load pretrained model
model.load_state_dict(torch.load(LOARD_MODEL))
model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)

if torch.cuda.is_available():
    model.cuda()

test_dataset = Dataset(file_list=TEST_FILE_LIST)
test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

# inference(test_loader, model, OUTPUT_DIR)
slide_inference(test_loader, model, OUTPUT_DIR)