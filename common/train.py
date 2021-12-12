import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import parse_args
from common.utils import get_optimizer, get_scheduler
from adv_lib.attack import attack_module
import models.classifier as C
from datasets import get_dataset
from utils.utils import load_checkpoint, set_random_seed

P = parse_args()

### Set torch device ###
if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU

set_random_seed(P.seed)

### Initialize dataset ###
train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, augment=P.augment_type)
P.image_size = image_size
P.n_classes = n_classes

### Define data loader ###
kwargs = {'pin_memory': True, 'num_workers': 8}
train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

if P.augment_type == 'autoaug_sche':
    train_set_second, _, _, _ = get_dataset(P, dataset=P.dataset, augment='autoaug')
    P.train_second_loader = DataLoader(train_set_second, shuffle=True, batch_size=P.batch_size, **kwargs)

### Initialize model ###
model = C.get_classifier(P, n_classes=P.n_classes).to(device)
optimizer, lr_decay_gamma = get_optimizer(P, model)
scheduler = get_scheduler(P, optimizer, lr_decay_gamma)

if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=not P.no_strict)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch']
    best = config['best']
    is_best = False
    error = 100.0
else:
    resume = False
    is_best = False
    start_epoch = 1
    best = 100.0
    error = 100.0

criterion = nn.CrossEntropyLoss().to(device)
adversary, adversary_t = attack_module(P, model, criterion)
