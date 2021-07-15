import time
from torch import nn
from math import log10


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


def train(current_epoch, total_epoch, trainloader, optimizer, model):
    model.train()
    avg_loss = AverageMeter()
    avg_psnr = AverageMeter()
    loss = nn.MSELoss()

    for i_iter, batch in enumerate(trainloader, 0):
        start_time = time.time()
        images, labels, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)
        losses = loss(preds, labels)
        psnr = 10 * log10(1 / losses.item())

        avg_loss.update(losses.item())
        avg_psnr.update(psnr)

        model.zero_grad()
        losses.backward()
        optimizer.step()

        if i_iter % 10 == 0:
            print(f"epcoh: {current_epoch}/{total_epoch}, iter:{i_iter}/{len(trainloader)}, time: {time.time()-start_time} , loss: {avg_loss.average()}, psnr: {avg_psnr.average()}")
