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


class MiniKPCNNSingleFrame(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_input_channels = 12
        self.hidden_input_size = 8
        self.num_preconv_kernls = 16
        self.preconv_kernel_size = 9

        print("Using lightweight duty model!")

        # initial feature extraction pass
        modules = [
            nn.Conv2d(self.num_input_channels + self.hidden_input_size, 16, 7, padding=3, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, self.hidden_input_size + self.num_preconv_kernls, 5, padding=2, padding_mode="reflect"),
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

        norm_factor = self.preconv_kernel_size * self.preconv_kernel_size
        self.dwkernel = nn.Parameter(
            (torch.ones(self.num_preconv_kernls, 1, self.preconv_kernel_size, self.preconv_kernel_size) +
             torch.randn(self.num_preconv_kernls, 1, self.preconv_kernel_size, self.preconv_kernel_size) * 0.1)
             / norm_factor
        )

    def forward(self, input):

        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        W = input.size(2)
        H = input.size(3)

        # color is channels 0..2
        color = input[:, 0:3, :, :]

        # dim is (B, N, W, H)
        # where N is the number of output convolutions
        # we want our dims to be (B, 3, N, W, H)
        raw_output = self.model(input)

        # update our hidden input
        hidden_input = raw_output[:, 0:self.hidden_input_size, :, :]

        kernel = self.smax(raw_output[:, self.hidden_input_size:, :, :] / 5) # idea suggested in softmax paper to increase convergence rates
        kernel = kernel.view(B, 1, -1, W, H).expand(-1, 3, -1, -1, -1)

        # we need to do the same preconv
        # we want the output to be also the same dimension as kernel
        rpkernel = self.dwkernel.repeat(3, 1, 1, 1)
        preconv = F.conv2d(color, rpkernel, stride=1, padding=self.preconv_kernel_size // 2, groups=3).view(B, 3, -1, W, H)
        # now each kernel has outputted N different channels
        # so our dimensions are now (B, 3, N, W, H)

        filtered = (kernel * preconv).sum(2)

        output = torch.concat((filtered, hidden_input), dim=1)

        return output

class MiniKPCNN2(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using double-frame model!")

        self.sf0 = MiniKPCNNSingleFrame()
        self.sf1 = MiniKPCNNSingleFrame()

        self.num_input_channels = 12
        self.hidden_input_size = 8
        self.num_preconv_kernls = 16
        self.preconv_kernel_size = 9

    def forward(self, input):

        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        W = input.size(2)
        H = input.size(3)


        hidden_input = torch.zeros(B, self.hidden_input_size, W, H, device=input.get_device())
        for i in range(num_frames):
            base_channel_index = i * self.num_input_channels


            frame_input = torch.concat(
                (input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :], hidden_input),
                dim=1
            )



            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]

            filtered0 = self.sf0(frame_input)

            # combine the output
            frame_input1 = torch.concat((filtered0[:, 0:3, :, :], frame_input[:, 3:self.num_input_channels], filtered0[:, 3:11, :, :]), dim=1)

            filtered1 = self.sf1(frame_input1)
            hidden_input = filtered1[:, 3:11, :, :]

            frame_remodulated = albedo * filtered1[:, 0:3, :, :]
            if i == 0:
                remodulated = frame_remodulated
                #saved_hidden_input = hidden_input[:, 0:3, :, :]
                saved_color = filtered0[:, 0:3, :, :]
            else:
                remodulated = torch.concat((remodulated, frame_remodulated), dim=1)
                #saved_hidden_input = torch.concat((saved_hidden_input, hidden_input[:, 0:3, :, :]), dim=1)
                saved_color = torch.concat((saved_color, filtered0[:, 0:3, :, :]), dim=1)

        return remodulated, saved_color

class MiniKPCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_input_channels = 12
        self.hidden_input_size = 8
        self.num_preconv_kernls = 16
        self.preconv_kernel_size = 9

        print("Using lightweight duty model!")

        # initial feature extraction pass
        modules = [
            nn.Conv2d(self.num_input_channels + self.hidden_input_size, 16, 7, padding=3, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, self.hidden_input_size + self.num_preconv_kernls, 5, padding=2, padding_mode="reflect"),
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

        norm_factor = self.preconv_kernel_size * self.preconv_kernel_size
        self.dwkernel = nn.Parameter(
            (torch.ones(self.num_preconv_kernls, 1, self.preconv_kernel_size, self.preconv_kernel_size) +
             torch.randn(self.num_preconv_kernls, 1, self.preconv_kernel_size, self.preconv_kernel_size) * 0.1)
             / norm_factor
        )

    def forward(self, input):

        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        W = input.size(2)
        H = input.size(3)


        hidden_input = torch.zeros(B, self.hidden_input_size, W, H, device=input.get_device())
        for i in range(num_frames):
            base_channel_index = i * self.num_input_channels

            frame_input = torch.concat(
                (input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :], hidden_input),
                dim=1
            )

            # color is channels 0..2
            color = frame_input[:, 0:3, :, :]
            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]

            # dim is (B, N, W, H)
            # where N is the number of output convolutions
            # we want our dims to be (B, 3, N, W, H)
            raw_output = self.model(frame_input)

            # update our hidden input
            hidden_input = raw_output[:, 0:self.hidden_input_size, :, :]

            kernel = self.smax(raw_output[:, self.hidden_input_size:, :, :] / 5) # idea suggested in softmax paper to increase convergence rates
            kernel = kernel.view(B, 1, -1, W, H).expand(-1, 3, -1, -1, -1)

            # we need to do the same preconv
            # we want the output to be also the same dimension as kernel
            rpkernel = self.dwkernel.repeat(3, 1, 1, 1)
            preconv = F.conv2d(color, rpkernel, stride=1, padding=self.preconv_kernel_size // 2, groups=3).view(B, 3, -1, W, H)
            # now each kernel has outputted N different channels
            # so our dimensions are now (B, 3, N, W, H)

            filtered = (kernel * preconv).sum(2)

            frame_remodulated = albedo * filtered
            if i == 0:
                remodulated = frame_remodulated
            else:
                remodulated = torch.concat((remodulated, frame_remodulated), dim=1)

        return remodulated


class DPKPCNNHybrid(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using double-frame model!")

        self.num_input_channels = 12
        self.hidden_input_size = 8

        self.sf0 = MiniKPCNNSingleFrame()

        # initial feature extraction pass
        modules = [
            nn.Conv2d(self.num_input_channels + self.hidden_input_size, 16, 7, padding=3, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, 3 + self.hidden_input_size, 5, padding=2, padding_mode="reflect"),
        ]

        self.model = nn.Sequential(
            *modules
        )



    def forward(self, input):

        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        W = input.size(2)
        H = input.size(3)


        hidden_input = torch.zeros(B, self.hidden_input_size, W, H, device=input.get_device())
        for i in range(num_frames):
            base_channel_index = i * self.num_input_channels

            frame_input = torch.concat(
                (input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :], hidden_input),
                dim=1
            )

            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]

            filtered0 = self.sf0(frame_input)

            # combine the output
            frame_input1 = torch.concat((filtered0[:, 0:3, :, :], frame_input[:, 3:self.num_input_channels], filtered0[:, 3:11, :, :]), dim=1)

            filtered1 = self.model(frame_input1)
            hidden_input = filtered1[:, 3:11, :, :]

            frame_remodulated = albedo * filtered1[:, 0:3, :, :]
            if i == 0:
                remodulated = frame_remodulated
            else:
                remodulated = torch.concat((remodulated, frame_remodulated), dim=1)

        return remodulated
