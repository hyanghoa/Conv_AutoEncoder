from torch import nn
from math import log10


def train(current_epoch, total_epoch, trainloader, optimizer, model):
    model.train()

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _ = batch
        size = images.size()
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        loss = nn.MSELoss()
        loss = loss(preds, labels)
        loss = loss.mean()
        psnr = 10 * log10(1 / loss.item())
        
        losses = []
        psnrs = []
        losses.append(loss)
        psnrs.append(psnr)
        # if dist.is_distributed():
        #     reduced_loss = reduce_tensor(loss)
        # else:
        #     reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i_iter%10 == 0:
            # print(f"epcoh: {current_epoch}/{total_epoch}, iter:{i_iter}/{len(trainloader)} loss: {loss}, psnr: {psnr}")
            if i_iter != 0:
                print(f"epcoh: {current_epoch}/{total_epoch}, iter:{i_iter}/{len(trainloader)} loss: {sum(losses)/i_iter}, psnr: {sum(psnrs)/i_iter}")

    #     # measure elapsed time
    #     batch_time.update(time.time() - tic)
    #     tic = time.time()

    #     # update average loss
    #     ave_loss.update(reduced_loss.item())

    #     lr = adjust_learning_rate(optimizer,
    #                               base_lr,
    #                               num_iters,
    #                               i_iter+cur_iters)

    #     if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
    #         msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
    #               'lr: {}, Loss: {:.6f}, PSNR: {:.4f}' .format(
    #                   epoch, num_epoch, i_iter, epoch_iters,
    #                   batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(), avg_psnr.average())
    #         logging.info(msg)

    # writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    # writer_dict['train_global_steps'] = global_steps + 1