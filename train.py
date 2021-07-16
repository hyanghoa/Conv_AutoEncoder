import time
from torch import nn
from math import log10


def train(current_epoch, total_epoch, train_loader, optimizer, scheduler, model, loss, log):
    model.train()
    avg_loss = AverageMeter()
    avg_psnr = AverageMeter()
    loss = nn.MSELoss()

    for i_iter, batch in enumerate(train_loader):
        start_time = time.time()
        images, labels, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)
        losses = loss(preds, labels)

        try:
            psnr = 10 * log10(2.64**2 / losses.item())
        except:
            log.write("zero division!")
            log.write(losses.item())
            log.write(losses)
            psnr = 10

        avg_loss.update(losses.item())
        avg_psnr.update(psnr)

        model.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()
        
        if i_iter % 10 == 0:
            log.write(f"epcoh: {current_epoch}/{total_epoch}, iter: {i_iter}/{len(train_loader)}, time: {(time.time()-start_time):.2f}, lr: {scheduler.get_last_lr()[0]}, loss: {avg_loss.average()}, psnr: {avg_psnr.average()}")

def val(val_loader, model, loss, log):
    model.eval()
    avg_loss = AverageMeter()
    avg_psnr = AverageMeter()
    loss = nn.MSELoss()

    for i_iter, batch in enumerate(val_loader):
        start_time = time.time()
        images, labels, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)
        losses = loss(preds, labels)
        
        try:
            psnr = 10 * log10(2.64**2 / losses.item())
        except:
            log.write("zero division!")
            psnr = 10

        avg_loss.update(losses.item())
        avg_psnr.update(psnr)

        if i_iter % 10 == 0:
            log.write(f"validate, iter: {i_iter}/{len(val_loader)}, time: {time.time()-start_time} , loss: {avg_loss.average()}, psnr: {avg_psnr.average()}")

    return avg_psnr.average()


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