import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best = None
        self.do_stop = False

    def __call__(self, metric=None, loss=None):

        assert metric is not None or loss is not None, "Either metric or loss should be provided"
        current = metric if metric is not None else -loss

        if self.best is None:
            self.best = current
        elif current < self.best + self.delta:
            self.counter += 1
            print(f"EarlyStopping: {current=} and {self.best=}. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.do_stop = True
        else:
            self.best = current
            self.counter = 0
