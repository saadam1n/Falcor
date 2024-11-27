import torch
import torch.nn as nn
import torch.nn.functional as F

class KPCNN(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using the super heavy duty model!")

        self.model = nn.Sequential(
            nn.Conv2d(12, 16, 25, padding=12, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 121, 11, padding=5, padding_mode="reflect"),
            nn.ReLU(),
        )

        # https://discuss.pytorch.org/t/initialising-weights-in-nn-sequential/76553
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.model.apply(weights_init)

        self.smax = nn.Softmax(dim=1)
        self.unfold = nn.Unfold(kernel_size=11, padding=5)

        self.preconv = nn.Conv2d(3, 25, 13, padding=6, padding_mode="reflect")

    def forward(self, input):

        W = input.size(2)
        H = input.size(3)

        # color is channels 0..2
        color = input[:, 0:3, :, :]
        # albedo is channels 3..5
        albedo = input[:, 3:6, :, :]


        kernel = self.smax(self.model(input) / 5) # idea suggested in softmax paper to increase convergence rates

        if True:
            # our dims are now (B, 25, W, H)
            # we want the same value per channel though
            kernel = kernel.view(-1, 1, 121, W, H).repeat(1, 3, 1, 1, 1)


            # after unfolding, our dim is (B, 3*5*5, num_blocks)
            # let's view it as (B, 3, 5*5, W, H)
            patches = self.unfold(color).view(-1, 3, 121, W, H)

            filtered = (kernel * patches).sum(2)
        else:
            # we are now (B, 25, W, H)
            patches = self.preconv(color)

            # kernel is also (B, 25, W, H)
            # we can just do multiplication and call it a day
            filtered = (kernel * patches).sum(1)


        remodulated = albedo * filtered
        return remodulated


class MiniKPCNN(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using lightweight duty model!")

        # initial feature extraction pass
        modules = [
            nn.Conv2d(12, 32, 7, padding=3, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(32, 8, 5, padding=2, padding_mode="reflect"),
        ]

        self.model = nn.Sequential(
            *modules
        )

        # https://discuss.pytorch.org/t/initialising-weights-in-nn-sequential/76553
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.model.apply(weights_init)

        self.smax = nn.Softmax(dim=1)

        self.dwkernel = nn.Parameter(torch.randn(8, 1, 7, 7))

    def forward(self, input):

        B = input.size(0)
        W = input.size(2)
        H = input.size(3)

        # color is channels 0..2
        color = input[:, 0:3, :, :]
        # albedo is channels 3..5
        albedo = input[:, 3:6, :, :]

        albedo[albedo < 0.001]=1.0
        color=color/albedo
        input[:, 0:3, :, :]=color
        input[:, 3:6, :, :]=albedo

        # dim is (B, N, W, H)
        # where N is the number of output convolutions
        # we want our dims to be (B, 3, N, W, H)
        kernel = self.smax(self.model(input) / 5) # idea suggested in softmax paper to increase convergence rates
        kernel = kernel.view(B, 1, -1, W, H).expand(-1, 3, -1, -1, -1)

        # we need to do the same preconv
        # we want the output to be also the same dimension as kernel
        rpkernel = self.dwkernel.repeat(3, 1, 1, 1)
        preconv = F.conv2d(color, rpkernel, stride=1, padding=3, groups=3).view(B, 3, -1, W, H)
        # now each kernel has outputted N different channels
        # so our dimensions are now (B, 3, N, W, H)

        filtered = (kernel * preconv).sum(2)

        remodulated = albedo * filtered
        return remodulated


