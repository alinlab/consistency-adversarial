import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_optimizer(P, model):
    if P.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
        lr_decay_gamma = 0.1
    else:
        raise NotImplementedError()

    return optimizer, lr_decay_gamma


def get_scheduler(P, optimizer, lr_decay_gamma):
    if P.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
    elif P.lr_scheduler == 'multi_step_decay':
        milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    else:
        raise NotImplementedError()

    return scheduler
