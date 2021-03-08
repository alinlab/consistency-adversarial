# The source code is from: https://github.com/yaodongyu/TRADES
import torch
import torch.nn.functional as F


def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), F.softmax(targets, dim=1),
                    reduction=reduction)


def _batch_l2norm(x):
    x_flat = x.view(x.size(0), -1)
    return torch.norm(x_flat, dim=1)


def generate_trades(model, x_natural, distance='Linf',
                    eps=0.031, eps_iter=0.003, nb_iter=10,
                    clip_min=0., clip_max=1.):
    is_training = model.training
    model.eval()

    # generate adversarial example
    x_adv = x_natural + 0.001 * torch.randn_like(x_natural)
    x_adv = x_adv.detach()

    outputs_natural = model(x_natural).detach()

    if distance == 'Linf':
        for _ in range(nb_iter):
            x_adv.requires_grad_()
            loss_kl = kl_div(model(x_adv), outputs_natural, reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + eps_iter * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - eps), x_natural + eps)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == 'L2':
        for _ in range(nb_iter):
            x_adv.requires_grad_()
            loss_kl = kl_div(model(x_adv), outputs_natural, reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

            x_adv = x_adv.detach() + eps_iter * grad
            eta_x_adv = x_adv - x_natural
            eta_x_adv = eta_x_adv.renorm(p=2, dim=0, maxnorm=eps)

            x_adv = x_natural + eta_x_adv
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        raise NotImplementedError()

    model.train(is_training)
    x_adv = x_adv.detach()

    return x_adv


def trades_loss(model, x_natural, y, distance='Linf',
                eps_iter=0.003, eps=0.031, nb_iter=10, beta=1.0,
                clip_min=0.0, clip_max=1.0):
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    if beta == 0:
        return loss_natural, 0

    x_adv = generate_trades(model, x_natural, distance=distance,
                            eps=eps, eps_iter=eps_iter, nb_iter=nb_iter,
                            clip_min=clip_min, clip_max=clip_max)
    loss_robust = kl_div(model(x_adv), model(x_natural))

    return loss_natural, beta * loss_robust
