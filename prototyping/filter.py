import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *
from bilateral_filter import BilateralFilter

def run_lcn(color, lcn_size):
    lcn_kernel = torch.ones(3, 1, lcn_size, lcn_size, device=color.device)
    lcn_padding = lcn_size // 2


    window_sum = F.conv2d(color, lcn_kernel, padding=lcn_padding, groups=3)
    window_square_sum = F.conv2d(color ** 2, lcn_kernel, padding=lcn_padding, groups=3)

    mean = window_sum / (lcn_size ** 2)
    std = window_square_sum - (mean ** 2) + 1e-5

    norm_color = (color - mean) / std

    return norm_color

class SemiFixedBilateralFilter(nn.Module):
    def __init__(self, total_dim, dynamic_size, kernel_size, dialation):
        super().__init__()

        self.kernel_size = kernel_size
        self.dialation = dialation
        self.dynamic_size = dynamic_size

        self.params = nn.Parameter(torch.randn(total_dim + 2))

        self.params.data.abs_().mul_(-0.01)

        self.clamp()

    def forward(self, input):
        # format: (dynamic dim, fixed dim)
        output = BilateralFilter.apply(input, self.params, self.kernel_size, self.dialation)

        # return only dynamic component
        return output[:, 0:self.dynamic_size, :, :]

    def clamp(self):
        with torch.no_grad():
            self.params.clamp_(max=0.0)

class WeightTransformBilateralFilter(nn.Module):
    def __init__(self, total_dim, dynamic_size, kernel_size, dialation):
        super().__init__()

        self.kernel_size = kernel_size
        self.dialation = dialation
        self.dynamic_size = dynamic_size

        self.params = nn.Parameter(torch.randn(total_dim + 2))

        # begin with overblurring
        self.use_exp = False

        if self.use_exp:
            self.params.data.abs_().mul_(-2.5)
        else:
            self.params.data.mul_(1.0) #0.2

    def forward(self, input):
        if self.use_exp:
            transformed_params = -torch.exp(self.params)
        else:
            transformed_params = -self.params * self.params

        # format: (dynamic dim, fixed dim)
        output = BilateralFilter.apply(input, transformed_params, self.kernel_size, self.dialation)

        # return only dynamic component
        return output[:, 0:self.dynamic_size, :, :]


# format: (RGB hidden fixed)
class ColorStableBilateralFilter(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, num_feature_channels, kernel_size, dilation):
        super().__init__()

        self.num_hidden_channels = num_hidden_channels

        # for dialation 1, we want padding 1, for dialation 2 we want padding 2, for dialtion 4 we want padding 4
        padding = 1#dilation

        total_input_channels = num_input_channels + self.num_hidden_channels

        """
        self.encoder = nn.Sequential(
            nn.Conv2d(total_input_channels, total_input_channels * 2, kernel_size=3, padding=padding, dilation=1),
            nn.ReLU(),
            nn.Conv2d(total_input_channels * 2, total_input_channels * 4, kernel_size=3, padding=padding, dilation=1),
            nn.ReLU(),
            nn.Conv2d(total_input_channels * 4, num_feature_channels , kernel_size=3, padding=padding, dilation=1)
        )
        """

        self.color_recombiner = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=padding, dilation=1),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=3, padding=padding, dilation=1),
            nn.ReLU(),
            nn.Conv2d(12, 3, kernel_size=3, padding=padding, dilation=1),
        )

        self.lcn_size = 5

        self.encoder = ConvEncoder(total_input_channels, num_feature_channels)
        self.filter = SemiFixedBilateralFilter(num_feature_channels + self.num_hidden_channels + 3, self.num_hidden_channels + 3, kernel_size, dilation)


    def forward(self, input):
        B = input.size(0)
        H = input.size(2)
        W = input.size(3)

        color = input[:, :3, :, :]
        norm_color = run_lcn(color, self.lcn_size)

        recombiner_input = torch.cat((norm_color, color), dim=1)
        recombined = self.color_recombiner(recombiner_input)

        enc_input = torch.cat((recombined, input[:, 3:, :, :]), dim=1)

        # extract features
        features = self.encoder(enc_input)

        combined = torch.cat((input[:, 0:3 + self.num_hidden_channels , :, :], features), dim=1)
        filtered = self.filter(combined)

        return filtered

    def clamp(self):
        self.filter.clamp()

# format: (RGB hidden fixed)
class AuxEncoderBilateralFilter(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, num_feature_channels, Encoder, kernel_size, dilation):
        super().__init__()

        self.num_hidden_channels = num_hidden_channels

        self.lcn_size = 5

        self.color_recombiner = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(12, 3, kernel_size=3, padding=1, dilation=1),
        )

        self.encoder = Encoder(num_input_channels, num_feature_channels, dilation)
        self.filter = WeightTransformBilateralFilter(num_feature_channels + self.num_hidden_channels + 3, self.num_hidden_channels + 3, kernel_size, dilation)

    def forward(self, input):
        B = input.size(0)
        H = input.size(2)
        W = input.size(3)

        color = input[:, :3, :, :]
        norm_color = run_lcn(color, self.lcn_size)

        recombiner_input = torch.cat((norm_color, color), dim=1)
        recombined = self.color_recombiner(recombiner_input)

        enc_input = torch.cat((recombined, input[:, 3:, :, :]), dim=1)

        # extract features
        features = self.encoder(enc_input)

        combined = torch.cat((input[:, 0:3 + self.num_hidden_channels , :, :], features), dim=1)
        filtered = self.filter(combined)

        return torch.cat((filtered, features), dim=1)



# format: (RGB hidden fixed)
class DownturnEncoderBilateralFilter(nn.Module):
    def __init__(self, num_passthrough_channels, num_hidden_channels, num_downturn_channels, Encoder0, Encoder1, kernel_size, dilation):
        super().__init__()

        self.num_hidden_channels = num_hidden_channels

        self.lcn_size = 5

        self.color_recombiner = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(12, 3, kernel_size=3, padding=1, dilation=1),
        )

        total_num_input_channels = num_passthrough_channels + self.num_hidden_channels + 3
        self.encoder0 = Encoder0(total_num_input_channels, num_passthrough_channels + self.num_hidden_channels, dilation)
        self.encoder1 = Encoder1(total_num_input_channels, num_downturn_channels, dilation)
        self.filter = WeightTransformBilateralFilter(num_downturn_channels + self.num_hidden_channels + 3, self.num_hidden_channels + 3, kernel_size, dilation)

    def forward(self, input):
        B = input.size(0)
        H = input.size(2)
        W = input.size(3)

        color = input[:, :3, :, :]
        norm_color = run_lcn(color, self.lcn_size)

        recombiner_input = torch.cat((norm_color, color), dim=1)
        recombined = self.color_recombiner(recombiner_input)

        enc_input = torch.cat((recombined, input[:, 3:, :, :]), dim=1)
        passthrough = self.encoder0(enc_input)

        features = self.encoder1(torch.cat((recombined, passthrough), dim=1))


        combined = torch.cat((input[:, 0:3 + self.num_hidden_channels , :, :], features), dim=1)
        filtered = self.filter(combined)

        combined_hidden = filtered[:, 3:, :, :] + passthrough[:, :self.num_hidden_channels, :, :]

        return torch.cat((filtered[:, :3, :, :], combined_hidden, passthrough[:, self.num_hidden_channels:, :, :]), dim=1)
