from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(
        description='PyTorch implementation of Consistency Regularization for Adversarial Robustness'
    )

    parser.add_argument('--dataset', help='Dataset',
                        type=str)
    parser.add_argument('--model', help='Model',
                        default='pre_resnet18', type=str)
    parser.add_argument('--mode', help='Training mode',
                        default='adv_train', type=str)

    parser.add_argument("--seed", type=int,
                        default=0, help='random seed')
    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')  # Currently not supported
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=1, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save checkpoint',
                        default=25, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=200, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd'],
                        default='sgd', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['multi_step_decay', 'cosine'],
                        default='multi_step_decay', type=str)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=5e-4, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)

    ##### Adv Training Configurations #####
    parser.add_argument('--epsilon', help='epsilon ball size',
                        default=None, type=float)
    parser.add_argument('--alpha', help='attack step size for iterative attacks',
                        default=None, type=float)
    parser.add_argument('--n_iters', help='number of iteration for iterative attacks',
                        default=None, type=int)
    parser.add_argument('--beta', help='regularization parameter for trades and mart',
                        default=6.0, type=float)
    parser.add_argument('--distance', help='adversarial attack ball type',
                        choices=['Linf', 'L2', 'L1'],
                        default='Linf', type=str)
    parser.add_argument('--adv_method', help='adversarial attack type',
                        choices=['fgsm', 'pgd'],
                        default='pgd', type=str)

    ##### Augmentation and Consistency Regularization Configurations #####
    parser.add_argument("--consistency", help='apply consistency regularization',
                        action='store_true')
    parser.add_argument('--augment_type', help='data augmentation type',
                        default='base', type=str)
    parser.add_argument('--lam', help='regularization parameter for consistency',
                        default=1.0, type=float)
    parser.add_argument('--T', help='temperature scaling',
                        default=0.5, type=float)

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()
