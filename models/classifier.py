from models.wide_resnet import wide_resnet_34_10
from models.resnet import pre_resnet18


def get_classifier(P, n_classes=10):
    if P.model == 'pre_resnet18':
        if P.dataset == 'tinyimagenet':
            classifier = pre_resnet18(num_classes=n_classes, stride=2)
        else:
            classifier = pre_resnet18(num_classes=n_classes)
    elif P.model == 'wrn3410':
        classifier = wide_resnet_34_10(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier
