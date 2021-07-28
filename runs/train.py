import time
import os
from math import log10

from tqdm import tqdm
import cv2
import torch
from torchvision import transforms, utils
# from utils.utils import AverageMeter


def train(current_epoch, total_epoch, train_loader, optimizer, scheduler, model, loss, l1_loss, scaler, log):
    model.train()
    avg_loss = AverageMeter()
    avg_psnr = AverageMeter()

    for i_iter, batch in enumerate(train_loader):
        optimizer.zero_grad()

        start_time = time.time()
        images, labels, _ = batch
        images = images.cuda()
        labels = labels.cuda()
        
        with torch.cuda.amp.autocast():
            preds = model(images)
            rmse_loss = torch.sqrt(loss(preds, labels))

        psnr = 20 * log10(1 / rmse_loss.item())
        avg_loss.update(rmse_loss.item())
        avg_psnr.update(psnr)

        # rmse_loss.backward()
        # optimizer.step()
        scaler.scale(rmse_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if i_iter % 100 == 0:
            for idx, p in enumerate(preds):
                # inv_normalize = transforms.Normalize(
                #     mean=[-0.176/0.222, -0.199/0.223, -0.189/0.225],
                #     std=[1/0.222, 1/0.223, 1/0.225]
                # )
                # inv_image = inv_normalize(p)
                # inv_image *= 255
                training_images = utils.make_grid(p*255).moveaxis(0, -1)
                cv2.imwrite(f"/home/djlee/autoencoder/outputs/effcientnet_unet_resize/training_images/train{idx}.png", cv2.cvtColor(training_images.float().detach().cpu().numpy(), cv2.COLOR_RGB2BGR))
        if i_iter % 10 == 0:
            log.write(
                f"epcoh: {current_epoch}/{total_epoch}, iter: {i_iter}/{len(train_loader)}, time: {(time.time()-start_time):.2f}, lr: {scheduler.get_last_lr()[0]}, loss: {avg_loss.average()}, psnr: {avg_psnr.average()}")


def val(val_loader, model, loss, log):
    model.eval()
    avg_loss = AverageMeter()
    avg_psnr = AverageMeter()

    for i_iter, batch in enumerate(val_loader):
        start_time = time.time()
        images, labels, _ = batch
        images = images.cuda()
        labels = labels.cuda()
        
        with torch.cuda.amp.autocast():
            preds = model(images)
            rmse_loss = torch.sqrt(loss(preds, labels))

        psnr = 20 * log10(1 / rmse_loss.item())

        avg_loss.update(rmse_loss.item())
        avg_psnr.update(psnr)

        if i_iter % 10 == 0:
            log.write(
                f"validate, iter: {i_iter}/{len(val_loader)}, time: {time.time()-start_time} , loss: {avg_loss.average()}, psnr: {avg_psnr.average()}")

    return avg_psnr.average()

def inference(test_loader, model, OUTPUT_DIR):
    model.eval()

    for i_iter, batch in enumerate(test_loader):
        start_time = time.time()
        images, names = batch
        with torch.cuda.amp.autocast():
            preds = model(images.cuda()).detach()
        inv_normalize = transforms.Normalize(
            mean=[-0.19317265/0.24399993, -0.21548774/0.25338732, -0.20241874/0.20241874],
            std=[1/0.24399993, 1/0.25338732, 1/0.20241874]
        )
        inv_images = inv_normalize(preds)
        inv_images *= 255
        for inv_image, name in zip(inv_images, names):
            inv_image = inv_image.moveaxis(0, -1).int().cpu().numpy()
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.png"), inv_image)

        print(f"iter: {i_iter}/{len(test_loader)}, time: {time.time()-start_time}")

def slide_inference(test_loader, model, OUTPUT_DIR):
    model.eval()

    for i_iter, batch in (enumerate(test_loader)):
        start_time = time.time()
        slide_size = 28
        images, names = batch
        patches = torch.zeros([images.size()[0], images.size()[1], images.size()[2], images.size()[3]])
        for h in tqdm(range(0, images.size()[2]-384, slide_size)):
            for w in tqdm(range(0, images.size()[3]-384, slide_size)):
                patch = images[:, :, h:h+384, w:w+384]
                with torch.cuda.amp.autocast():
                    pred = model(patch.cuda()).detach()
                    patches[:, :, h:h+384, w:w+384] = pred

        patches *= 255
        patches = torch.moveaxis(patches, 1, -1)
        for p, name in zip(patches, names):
            name = name.replace("_input", "")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.png"), cv2.cvtColor(p.numpy(), cv2.COLOR_BGR2RGB))

        print(f"iter: {i_iter}/{len(test_loader)}, time: {time.time()-start_time}")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg