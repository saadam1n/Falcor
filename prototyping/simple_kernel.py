import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleKernel(nn.Module):
    def __init__(self):
        super().__init__()

        # RGB in, RGB out, 7x7 kernel
        self.conv1 = nn.Conv2d(3, 3, 21, padding=10)
        nn.init.constant_(self.conv1.weight, 1.0 / (21.0 * 21.0))

    def forward(self, input):
        return self.conv1(input)


