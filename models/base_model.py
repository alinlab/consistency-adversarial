from abc import *
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(last_dim, num_classes)

    @abstractmethod
    def penultimate(self, inputs):
        pass

    def forward(self, inputs, penultimate=False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)
        output = self.linear(features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if _return_aux:
            return output, _aux

        return output
