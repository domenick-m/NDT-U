import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
import math

# from 

def get_norm(config, n_neurons):
    ''' Gets the normalization.

    Args:
        n_neurons (int): Number of neurons.
        config (dict): A config object.
    '''
    if config['model']['norm'] == 'layer':
        return nn.LayerNorm(n_neurons)
    elif config['model']['norm'] == 'switch':
        return SwitchNorm2d(n_neurons)


def get_optimizer(model, config):
    ''' Gets the optimizer.

    Args:
        model (Transformer): The model.
        config (dict): A config object.
    '''
    parameters = model.parameters()
    if config.train.optimizer == 'AdamW':
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, parameters), 
            lr=config.train.init_lr,
            weight_decay=config.train.weight_decay)
            
    # elif config['train']['optimizer'] == 'new optimizer':
    #     return torch.optim.AdamW(parameters,)
    

def get_scheduler(optimizer, config):
# def get_scheduler(optimizer, config, dataloader_size):
    ''' Gets the scheduler.

    Args:
        optimizer (optimizer): The optimizer to schedule.
        config (dict): A config object.
    '''
    scheduler = None
    if config['train']['scheduler'] == 'Cosine':
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=config['train']['warmup_steps'],
            t_total=config['train']['epochs']
        )
    # elif config['train']['scheduler'] == 'new scheduler':
    #     scheduler = new scheduler(optimizer,)
    return scheduler


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        x = x.permute(1, 2, 0)
        N, C, H = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H)
        x = x * self.weight + self.bias
        return x.permute(2, 0, 1)


class WarmupCosineSchedule(LambdaLR):
    '''WarmupCosineSchedule'''
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
