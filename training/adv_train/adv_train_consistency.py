import time

import torch.optim

from training import _jensen_shannon_div
from utils.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary, logger=None):

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['con'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images[0].size(0)
        labels = labels.to(device)

        images_aug1, images_aug2 = images[0].to(device), images[1].to(device)
        images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B
        images_adv = adversary(images_pair, labels.repeat(2))

        outputs_adv = model(images_adv)
        loss_ce = criterion(outputs_adv, labels.repeat(2))

        ### consistency regularization ###
        outputs_adv1, outputs_adv2 = outputs_adv.chunk(2)
        loss_con = P.lam * _jensen_shannon_div(outputs_adv1, outputs_adv2, P.T)

        ### total loss ###
        loss = loss_ce + loss_con

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['cls'].update(loss_ce.item(), batch_size)
        losses['con'].update(loss_con.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossCon %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['con'].value))

        check = time.time()

    if P.optimizer == 'sgd':
        scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossCon %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['con'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_con', losses['con'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
