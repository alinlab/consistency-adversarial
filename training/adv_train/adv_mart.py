import time

import torch.optim

from adv_lib.mart import mart_loss
from utils.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary=None, logger=None):

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['mrt'] = AverageMeter()
    losses['adv'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        loss_adv, loss_mart = mart_loss(model, images, labels, optimizer, distance=P.distance,
                                        eps_iter=P.alpha, eps=P.epsilon, nb_iter=P.n_iters,
                                        beta=P.beta, clip_min=0, clip_max=1)
        loss = loss_mart + loss_adv
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['mrt'].update(loss_mart.item(), batch_size)
        losses['adv'].update(loss_adv.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossMrt %f] [LossAdv %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['mrt'].value, losses['adv'].value))

        check = time.time()

    if P.optimizer == 'sgd':
        scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossMrt %f] [LossAdv %f]' %
         (batch_time.average, data_time.average,
          losses['mrt'].average, losses['adv'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_mart', losses['mrt'].average, epoch)
        logger.scalar_summary('train/loss_adversary', losses['adv'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
