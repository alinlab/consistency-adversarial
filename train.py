from utils.utils import Logger
from utils.utils import save_checkpoint, save_checkpoint_epoch

from common.train import *
from evals import test_classifier_adv


kwargs = {}
if 'adv' in P.mode:
    from training.adv_train import setup
    kwargs['adversary'] = adversary
else:
    from training.train import setup

train, fname = setup(P.mode, P)

logger = Logger(fname, ask=not resume)
logger.log(P)
logger.log(model)

# Run experiments
for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    train(P, epoch, model, criterion, optimizer, scheduler, train_loader, logger=logger, **kwargs)
    model.eval()

    if epoch % P.error_step == 0:
        error = test_classifier_adv(P, model, test_loader, epoch,
                                    adversary=adversary_t, logger=logger, ret='adv')

        is_best = (best > error)
        if is_best:
            best = error

        logger.scalar_summary('eval/best_adv_error', best, epoch)
        logger.log('[Epoch %3d] [Adv_Test %5.2f] [Best %5.2f]' % (epoch, error, best))

    save_states = model.state_dict()
    save_checkpoint(epoch, best, save_states, optimizer.state_dict(), logger.logdir, is_best)
    if epoch % P.save_step == 0:
        save_checkpoint_epoch(epoch, save_states, optimizer.state_dict(), logger.logdir)
