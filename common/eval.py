import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import parse_args
from adv_lib.attack import attack_module
import models.classifier as C
from datasets import get_dataset
from utils.utils import set_random_seed

P = parse_args()

### Set torch device ###

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Initialize dataset ###
train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset)

P.image_size = image_size
P.n_classes = n_classes

set_random_seed(P.seed)

kwargs = {'pin_memory': False, 'num_workers': 4}
train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

if P.mode == 'test_mce':
    from datasets.datasets import get_cifar_c
    dataset_list, corruption_list = get_cifar_c(P.dataset)

    corruption_loader = dict()
    for i in range(len(corruption_list)):
        corruption_loader[corruption_list[i]] = DataLoader(dataset_list[i], shuffle=False,
                                                           batch_size=P.test_batch_size, **kwargs)

### Initialize model ###
model = C.get_classifier(P, n_classes=P.n_classes).to(device)

if P.load_path is not None:
    checkpoint = torch.load(P.load_path)
    model.load_state_dict(checkpoint, strict=not P.no_strict)

criterion = nn.CrossEntropyLoss().to(device)

adversary, adversary_t = attack_module(P, model, criterion, _eval=True)
