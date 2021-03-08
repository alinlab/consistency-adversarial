import torch

from utils.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results


def test_classifier(P, model, loader, steps, logger=None):
    error_top1 = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        batch_size = images.size(0)

        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)

        top1, = error_k(outputs.data, labels, ks=(1,))
        error_top1.update(top1.item(), batch_size)

    log_(' * [Error@1 %.3f] [Acc %.3f]' %
         (error_top1.average, 100. - error_top1.average))

    if logger is not None:
        logger.scalar_summary('eval/clean_error', error_top1.average, steps)

    model.train(mode)

    return error_top1.average


def test_classifier_adv(P, model, loader, steps, adversary=None,
                        logger=None, softmax=False, ret='clean'):
    error_top1 = AverageMeter()
    error_adv_top1 = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if adversary is None:
        adversary = lambda x, y: x

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        adv_images = adversary(images, labels)

        outputs = model(images)
        adv_outputs = model(adv_images)

        # Measure accuracy and record loss
        top1, = error_k(outputs.data, labels, ks=(1,))
        adv_top1, = error_k(adv_outputs.data, labels, ks=(1,))

        batch_size = images.size(0)
        error_top1.update(top1.item(), batch_size)
        error_adv_top1.update(adv_top1.item(), batch_size)

    log_(' * [Error@1 %.3f] [AdvError@1 %.3f]' %
         (error_top1.average, error_adv_top1.average))
    log_(' * [Acc@1 %.3f] [AdvAcc@1 %.3f]' %
         (100. - error_top1.average, 100. - error_adv_top1.average))

    if logger is not None:
        logger.scalar_summary('error/clean', error_top1.average, steps)
        logger.scalar_summary('error/adv', error_adv_top1.average, steps)

    model.train(mode)

    if ret == 'clean':
        return error_top1.average
    elif ret == 'adv':
        return error_adv_top1.average
    else:
        raise NotImplementedError()
