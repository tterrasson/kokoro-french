import torch


class MyDataParallel(torch.nn.DataParallel):
    """DataParallel subclass that forwards attribute access to the wrapped module."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
