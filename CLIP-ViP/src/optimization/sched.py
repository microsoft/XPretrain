"""
optimizer learning rate scheduling helpers
"""
import math
from math import ceil
from collections import Counter


def noam_schedule(step, warmup_step=4000):
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))

def warmup_cosine(step, warmup_step, tot_step):
    if step < warmup_step:
        return step / warmup_step
    progress = (step - warmup_step) / (tot_step - warmup_step)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def multi_step_schedule(n_epoch, milestones, step, warmup_step,gamma=0.5):
    if step <= warmup_step:
        return step / warmup_step

    milestones = list(sorted(milestones))
    for i, m in enumerate(milestones):
        if n_epoch < m:
            return gamma**i
    return gamma**(len(milestones)+1)

class AutoStep():
    def __init__(self, tolerance, gamma):
        self.tolerance = tolerance
        self.coeff_mem = 1
        self.gamma = gamma
        self.best_score = 0.
        self.count = 0

    def step(self, score):
        if score <= self.best_score:
            self.count += 1
        else:
            self.count = 0
        self.best_score = score
        if self.count > self.tolerance:
            self.count = 0
            self.coeff_mem = self.coeff_mem * self.gamma

    def get_lr(self, global_step, learning_rate, num_train_steps, warmup_ratio=0.1):
        warmup_steps = int(warmup_ratio * num_train_steps)
        if global_step <= warmup_steps:
            return learning_rate * global_step / warmup_steps

        return max(self.coeff_mem * learning_rate, 1e-8)


def get_lr_sched(global_step, decay, learning_rate,
                 num_train_steps, warmup_ratio=0.1,
                 decay_epochs=[], multi_step_epoch=-1):
    warmup_steps = int(warmup_ratio*num_train_steps)
    if decay == 'linear':
        lr_this_step = learning_rate * warmup_linear(
            global_step, warmup_steps, num_train_steps)
    elif decay == 'cosine':
        lr_this_step = learning_rate * warmup_cosine(
            global_step, warmup_steps, num_train_steps)
    elif decay == 'invsqrt':
        lr_this_step = learning_rate * noam_schedule(
            global_step, warmup_steps)
    elif decay == 'constant':
        lr_this_step = learning_rate
    elif decay == "multi_step":
        assert multi_step_epoch >= 0
        lr_this_step = learning_rate * multi_step_schedule(
            multi_step_epoch, decay_epochs, global_step, warmup_steps)
    if lr_this_step <= 0:
        # save guard for possible miscalculation of train steps
        lr_this_step = 1e-8
    return lr_this_step
