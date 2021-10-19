import numpy as np
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, initial_lr, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.initial_lr = initial_lr

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.initial_lr + self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Opt:
    "Optim wrapper that implements rate."

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        self.optimizer.step()

if __name__ == '__main__':
    opts = [NoamOpt(512, 1, 4000, 0.01, None),
            NoamOpt(512, 1, 8000, 0.01, None),
            NoamOpt(256, 1, 4000, 0.01, None)]