import os
import pickle
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, ask=True):
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        logdir = self._make_dir(fn)
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        if len(os.listdir(logdir)) != 0 and ask:
            ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                        "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)

        self.set_dir(logdir)

    def _make_dir(self, fn):
        today = datetime.today().strftime("%y%m%d")
        logdir = 'logs/' + today + '_' + fn
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins='auto')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def load_checkpoint(logdir, mode='last'):
    if mode == 'last':
        model_path = os.path.join(logdir, 'last.model')
        optim_path = os.path.join(logdir, 'last.optim')
        config_path = os.path.join(logdir, 'last.config')
    elif mode == 'best':
        model_path = os.path.join(logdir, 'best.model')
        optim_path = os.path.join(logdir, 'best.optim')
        config_path = os.path.join(logdir, 'best.config')

    else:
        raise NotImplementedError()

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        optim_state = torch.load(optim_path)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None

    return model_state, optim_state, cfg


def save_checkpoint(epoch, best, model_state, optim_state, logdir, is_best):
    last_model = os.path.join(logdir, 'last.model')
    best_model = os.path.join(logdir, 'best.model')
    last_optim = os.path.join(logdir, 'last.optim')
    best_optim = os.path.join(logdir, 'best.optim')
    last_config = os.path.join(logdir, 'last.config')
    best_config = os.path.join(logdir, 'best.config')

    opt = {
        'epoch': epoch,
        'best': best
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if is_best:
        shutil.copyfile(last_model, best_model)
        shutil.copyfile(last_optim, best_optim)
        shutil.copyfile(last_config, best_config)


def save_checkpoint_epoch(epoch, model_state, optim_state, logdir):
    last_model = os.path.join(logdir, f'epoch{epoch}.model')
    last_optim = os.path.join(logdir, f'epoch{epoch}.optim')
    last_config = os.path.join(logdir, f'epoch{epoch}.config')

    opt = {
        'epoch': epoch,
        'best': None
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def extract_dataset(loader):
    x_list, y_list = [], []
    for i, (x, y) in enumerate(loader):
        x_list.append(x)
        y_list.append(y)
    x_list = torch.cat(x_list)
    y_list = torch.cat(y_list)

    return x_list, y_list
