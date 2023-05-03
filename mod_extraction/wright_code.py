"""Code taken from https://github.com/Alec-Wright/CoreAudioML/"""
import logging
import os
from typing import List

import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class WrightESRLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.epsilon = 0.00001
        self.epsilon = 0.0  # Make it the same as auraloss

    def forward(self, output: T, target: T) -> T:
        loss = tr.add(target, -output)
        loss = tr.pow(loss, 2)
        loss = tr.mean(loss)
        energy = tr.mean(tr.pow(target, 2)) + self.epsilon
        loss = tr.div(loss, energy)
        return loss


class WrightDCLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.epsilon = 0.00001
        self.epsilon = 0.0  # Make it the same as auraloss

    def forward(self, output: T, target: T) -> T:
        loss = tr.pow(tr.add(tr.mean(target, 0), -tr.mean(output, 0)), 2)
        loss = tr.mean(loss)
        energy = tr.mean(tr.pow(target, 2)) + self.epsilon
        loss = tr.div(loss, energy)
        return loss


# PreEmph is a class that applies an FIR pre-emphasis filter to the signal, the filter coefficients are in the
# filter_cfs argument, and lp is a flag that also applies a low pass filter
# Only supported for single-channel!
class WrightPreEmph(nn.Module):
    def __init__(self, filter_cfs: List[float], low_pass: bool = False) -> None:
        super().__init__()
        self.epsilon = 0.00001
        self.zPad = len(filter_cfs) - 1

        self.conv_filter = nn.Conv1d(1, 1, kernel_size=(2,), bias=False)
        self.conv_filter.weight.data = tr.tensor([[filter_cfs]], requires_grad=False)

        self.low_pass = low_pass
        if self.low_pass:
            self.lp_filter = nn.Conv1d(1, 1, kernel_size=(2,), bias=False)
            self.lp_filter.weight.data = tr.tensor([[[0.85, 1]]], requires_grad=False)

    def forward(self, output: T, target: T) -> (T, T):
        # zero pad the input/target so the filtered signal is the same length
        output = tr.cat((tr.zeros(self.zPad, output.shape[1], 1), output))
        target = tr.cat((tr.zeros(self.zPad, target.shape[1], 1), target))
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        output = self.conv_filter(output.permute(1, 2, 0))
        target = self.conv_filter(target.permute(1, 2, 0))

        if self.low_pass:
            output = self.lp_filter(output)
            target = self.lp_filter(target)

        return output.permute(2, 0, 1), target.permute(2, 0, 1)
