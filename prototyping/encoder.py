import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, dilation=None):
        super().__init__()

        if type(dilation) is not None:
            print("WARNING: EncoderUNet does not take a dilation argument!")

        internal_channels = num_input_channels
        self.conv0 = nn.Sequential(
            nn.Conv2d(num_input_channels, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels * 2, kernel_size=3, padding=1),
        )
        self.pool0 = nn.MaxPool2d(2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, internal_channels * 4, kernel_size=3, padding=1),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(internal_channels * 4, internal_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 8, internal_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 8, internal_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 8, internal_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 8, internal_channels * 4, kernel_size=3, padding=1),
        )

        self.upscale0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(internal_channels * 8, internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 4, internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 4, internal_channels * 2, kernel_size=3, padding=1),
        )
        self.relu0 = nn.ReLU()

        self.upscale1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(internal_channels * 4, internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, num_output_channels, kernel_size=3, padding=1),
        )

    def forward(self, input):
        input = self.conv0(input)

        skip0 = input

        input = self.pool0(input)
        input = self.conv1(input)

        skip1 = input

        input = self.pool1(input)
        input = self.conv2(input)
        input = self.upscale0(input)

        input = torch.cat((input, skip1), dim=1)
        input = self.conv3(input)
        input = self.relu0(input)

        input = self.upscale1(input)
        input = torch.cat((input, skip0), dim=1)

        input = self.conv4(input)

        return input

class EncoderUNet2(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, dilation=None):
        super().__init__()

        if type(dilation) is not None:
            print("WARNING: EncoderUNet2 does not take a dilation argument!")

        internal_channels = num_output_channels
        self.down0 = nn.Sequential(
            nn.Conv2d(num_input_channels, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels * 2, kernel_size=3, padding=1),
        )

        self.pool0 = nn.MaxPool2d(2)

        self.enc = nn.Sequential(
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.upsample0 = nn.ConvTranspose2d(internal_channels * 2, internal_channels * 2, stride=2, kernel_size=2)

        self.up0 = nn.Sequential(
            nn.Conv2d(internal_channels * 2, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, num_output_channels, kernel_size=3, padding=1),
        )





    def forward(self, input):
        input = self.down0(input)

        skip0 = input

        input = self.pool0(input)
        input = self.enc(input)

        input = self.upsample0(input)
        input = input + skip0

        input = self.up0(input)

        return input

class ConvEncoder(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()

        internal_channels = num_output_channels * 2
        self.enc = nn.Sequential(
            nn.Conv2d(num_input_channels, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, num_output_channels, kernel_size=3, padding=1)
        )

    def forward(self, input):
        return self.enc(input)

class ConvEncoder2(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, dilation):
        super().__init__()

        internal_channels = num_output_channels
        padding = dilation
        self.enc = nn.Sequential(
            nn.Conv2d(num_input_channels, internal_channels * 2, kernel_size=3, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, num_output_channels, kernel_size=3, dilation=dilation, padding=padding),
        )

    def forward(self, input):
        return self.enc(input)


class ConvEncoder3(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, dilation):
        super().__init__()

        internal_channels = num_output_channels
        padding = dilation
        self.enc = nn.Sequential(
            nn.Conv2d(num_input_channels, internal_channels * 2, kernel_size=3, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, num_output_channels, kernel_size=3, dilation=dilation, padding=padding)
        )

    def forward(self, input):
        return self.enc(input)

# passthrough network
class ConvEncoder4(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, dilation):
        super().__init__()

        internal_channels = num_output_channels
        padding = dilation
        self.enc = nn.Sequential(
            nn.Conv2d(num_input_channels, internal_channels, kernel_size=3, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.Conv2d(internal_channels, num_output_channels, kernel_size=3, dilation=dilation, padding=padding),
            nn.ReLU(), # extra nonlinearity
        )

    def forward(self, input):
        return self.enc(input)

# downturn network
class ConvEncoder5(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, dilation):
        super().__init__()

        internal_channels = num_output_channels
        padding = dilation
        self.enc = nn.Sequential(
            nn.Conv2d(num_input_channels, internal_channels, kernel_size=3, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.Conv2d(internal_channels, num_output_channels, kernel_size=3, dilation=dilation, padding=padding)
        )

    def forward(self, input):
        return self.enc(input)


class DenoiseUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, dilation=None):
        super().__init__()

        if type(dilation) is not None:
            print("WARNING: Denoise U-Net does not take a dilation argument!")

        internal_channels = num_input_channels
        self.down0 = nn.Sequential(
            nn.Conv2d(num_input_channels, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels * 2, kernel_size=3, padding=1),
        )

        self.pool0 = nn.MaxPool2d(2)

        self.enc = nn.Sequential(
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(internal_channels * 2, internal_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.upsample0 = nn.Sequential(
            nn.ConvTranspose2d(internal_channels * 2, internal_channels * 2, stride=2, kernel_size=2),
            nn.ReLU()
        )

        self.up0 = nn.Sequential(
            nn.Conv2d(internal_channels * 2, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(internal_channels, num_output_channels, kernel_size=3, padding=1),
        )

        self.up0[3].weight.data.zero_()

    def forward(self, input):
        input = self.down0(input)

        skip0 = input

        input = self.pool0(input)
        input = self.enc(input)

        input = self.upsample0(input)
        input = input + skip0

        input = self.up0(input)

        return input

class VHEEncoder(nn.Module):
    def __init__(self, num_hidden_channels, kernel_size):
        super().__init__()

        total_input_channels = num_hidden_channels + 3
        padding = kernel_size // 2

        self.estimator = nn.Sequential(
            nn.Conv2d(2 * total_input_channels, 2 * total_input_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(2 * total_input_channels, 2 * total_input_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(2 * total_input_channels, num_hidden_channels + 1, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, input):
        input_squared = input * input

        enc_input = torch.cat((input, input_squared), dim=1)

        enc_output = self.estimator(enc_input)

        mean = enc_output[:, 0:1, :, :]
        norm_color = input[:, :3, :, :]# - mean

        hidden_state = enc_output[:, 1:, :, :]

        output = torch.cat((norm_color, hidden_state), dim=1)

        return output

# assumes input channels = output channels
class EESPEncoderSimple(nn.Module):
    # num_input_channels = all incoming channels, num_hidden_channels = num hidden channels within input,
    def __init__(self, num_input_channels, num_hidden_channels, num_groups, dilation):
        super().__init__()

        if type(dilation) is not None:
            print("WARNING: EESP Encoder does not take a dilation argument!")


        self.num_groups = num_groups
        self.num_channels_per_group = num_input_channels // self.num_groups

        if(num_input_channels % self.num_groups != 0):
            raise RuntimeError(f"Num groups was {num_groups} but input channels was the nondivisible number {num_input_channels}")

        self.num_hidden_channels = num_hidden_channels

        self.vhe = VHEEncoder(num_hidden_channels, 5)

        # 1x1 kernel to transform everything
        # this will be the expensive part of our network
        self.ld_encoder = nn.Sequential(
            nn.Conv2d(num_input_channels + 3, num_input_channels, kernel_size=1),
            nn.ReLU()
        )

        self.lane_encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1, groups=self.num_groups),
            nn.ReLU(),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1, groups=self.num_groups),
        )

        # I call it linear encoder because it acts like a linear layer
        self.linear_encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=1, groups=self.num_groups),
            nn.ReLU(),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=1, groups=self.num_groups),
        )

    def forward(self, input):
        B = input.size(0)
        H = input.size(2)
        W = input.size(3)

        norm_info = self.vhe(input[:, :3 + self.num_hidden_channels, :, :])
        input = torch.cat((norm_info, input[:, 3 + self.num_hidden_channels:, :, :]), dim=1)

        ld = self.ld_encoder(input)

        output = self.lane_encoder(ld)

        # reorganize our channels so our highest hidden channels are now at the bottom
        output = torch.cat((output[:, -self.num_hidden_channels:, :, :], output[:, :-self.num_hidden_channels, :, :]), dim=1)

        # create cumulative sum across each lane
        output = output.view(B, self.num_groups, self.num_channels_per_group, H, W)
        output = output.cumsum(1)
        output = output.view(B, self.num_groups * self.num_channels_per_group, H, W)

        output = self.linear_encoder(output)

        output = output + ld # use ld as our skip connection

        return output


class ColorStabilizingPool(nn.Module):
    # num_input_channels = all incoming channels, num_hidden_channels = num hidden channels within input,
    def __init__(self, num_avg_channels):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(2)
        self.max_pool = nn.MaxPool2d(2)

        self.avg_channels = num_avg_channels

    def forward(self, input):

        ares = self.avg_pool(input[:, :self.avg_channels, :, :])
        mres = self.max_pool(input[:, self.avg_channels:, :, :])

        output = torch.cat((ares, mres), dim=1)

        return output


