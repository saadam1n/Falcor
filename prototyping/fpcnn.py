import torch
import torch.nn as nn
from bilateral_filter import BilateralFilter

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

class SimpleBilateralFilter(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using simple bilateral filter")

        self.num_input_channels = 12

        self.params = nn.Parameter(torch.randn(self.num_input_channels + 2 + 14))

        self.params.data.abs_().mul_(-0.01)
        if False:
            # reduce emphasis on raw color and albedo for filtering for filtering
            self.params.data[0:6].mul_(0.2)
            self.params.data[14:20].mul_(0.2)
            # emphasize worldpos and worldnorm and distance
            self.params.data[6:14].mul_(4.0)
            self.params.data[20:28].mul_(4.0)
        else:
            # begin with box blur
            self.params.data.mul_(0.0)

        self.clamp()

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            # color is channels 0..2
            color = frame_input[:, 0:3, :, :]
            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]

            frame_output = BilateralFilter.apply(frame_input, self.params[0:14], 17, 1)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = albedo * frame_output[:, 0:3, :, :]

        return output

    def clamp(self):
        with torch.no_grad():
            self.params.clamp_(max=0.0)

class SomewhatComplexBilateralFilter(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using somewhat complex bilateral filter")

        self.num_input_channels = 12
        self.dynamic_size = 4

        tot_input_channels = self.num_input_channels + self.dynamic_size - 3

        self.bfilter = SemiFixedBilateralFilter(tot_input_channels, self.dynamic_size, 7, 1)

        self.fixed_encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, tot_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_input_channels, self.num_input_channels - 3, kernel_size=3, padding=1),
            #nn.Conv2d(self.num_input_channels, self.num_input_channels - 3, kernel_size=5, padding=2),
        )

        self.dyn_encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, tot_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_input_channels, self.dynamic_size, kernel_size=3, padding=1),
            #nn.Conv2d(self.num_input_channels, self.dynamic_size, kernel_size=5, padding=2),
        )

        self.decoder = nn.Sequential(
            #nn.Conv2d(tot_input_channels, tot_input_channels, kernel_size=3, padding=1),
            #nn.ReLU(),
            nn.Conv2d(tot_input_channels, 3, kernel_size=3, padding=1),
        )


        #nn.Conv2d(tot_input_channels, 3, kernel_size=1, padding=0)

        self.clamp()

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]
            fixed_component = self.fixed_encoder(frame_input) #frame_input[:, 3:12, :, :]

            enc = self.dyn_encoder(frame_input)

            extended_state = torch.cat((enc, fixed_component), dim=1)
            extended_state = self.bfilter(extended_state)
            extended_state = torch.cat((extended_state, fixed_component), dim=1)

            frame_output = self.decoder(extended_state)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = albedo * frame_output

        return output

    def clamp(self):
        self.bfilter.clamp()

class SimpleFPCNN(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using FPCNN")

        self.num_input_channels = 12
        self.dynamic_size = 6
        self.fixed_size = 10
        self.num_filters = 4

        tot_proc_channels = self.fixed_size + self.dynamic_size

        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, tot_proc_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_proc_channels, tot_proc_channels, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(tot_proc_channels, tot_proc_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_proc_channels, 3, kernel_size=3, padding=1),
        )

        self.filter_weight_predictor = nn.Sequential(
            nn.Conv2d(tot_proc_channels, tot_proc_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(tot_proc_channels, self.num_filters, kernel_size=5, padding=2),
            nn.Softmax(dim=1)
        )

        self.bfilters = nn.ModuleList()

        for i in range(self.num_filters):
            self.bfilters.add_module(str(i), SemiFixedBilateralFilter(tot_proc_channels, self.dynamic_size, 7, 1))

        self.clamp()

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]
            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]

            high_dim_space = self.encoder(frame_input)
            fixed = high_dim_space[:, -self.fixed_size:, :, :]

            predictions = self.filter_weight_predictor(high_dim_space)

            high_dim_filtered = torch.stack([filter(high_dim_space) for filter in self.bfilters], dim=1)

            fixed_expanded = fixed.unsqueeze(1).expand(-1, len(self.bfilters), -1, -1, -1)
            combined = torch.cat((high_dim_filtered, fixed_expanded), dim=2)

            low_dim_filtered = self.decoder(combined.view(-1, combined.shape[2], *combined.shape[3:]))
            low_dim_filtered = low_dim_filtered.view(high_dim_filtered.shape[0], len(self.bfilters), -1, *high_dim_filtered.shape[3:])

            filtered_responses = predictions.unsqueeze(2) * low_dim_filtered
            final_filtered = torch.sum(filtered_responses, dim=1)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = albedo * final_filtered

        return output

    def clamp(self):
        for i, filter in enumerate(self.bfilters):
            filter.clamp()

# this is part of my experiments to find the optimal encoder-decoder architecture for a bilateral filter
class ExtendedBilateralFilter(nn.Module):
    def __init__(self, num_input_channels, dynamic_size, fixed_size):
        super().__init__()

        # bilateral filters work on a rule-out basis
        # thus, they are better served with fewer but more meaningful features
        # if we have too many features it will be too easy to rule out a lot of the neighboring pixels

        self.num_input_channels = num_input_channels
        self.dynamic_size = dynamic_size
        self.fixed_size = fixed_size

        tot_channels = self.dynamic_size + self.fixed_size

        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, tot_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_channels, tot_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_channels, tot_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_channels, tot_channels, kernel_size=3, padding=1)
        )

        self.reconstruction_encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, tot_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_channels, tot_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_channels, tot_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_channels, self.fixed_size, kernel_size=3, padding=1)
        )

        self.filter = SemiFixedBilateralFilter(tot_channels, self.dynamic_size, kernel_size=7, dialation=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(tot_channels, tot_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_channels, tot_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tot_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]
            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]

            hd_space = self.encoder(frame_input)
            filtered_hd_space = self.filter(hd_space)

            reconstruction_guide = self.reconstruction_encoder(frame_input)

            recombined_hd_space = torch.cat((filtered_hd_space, reconstruction_guide), dim=1)

            ld_space = self.decoder(recombined_hd_space)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = albedo * ld_space

        return output

    def clamp(self):
        self.filter.clamp()


class SimpleFPCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using FPCNN 2")

        self.num_input_channels = 12
        self.dynamic_size = 8
        self.fixed_size = 8

        self.filter = ExtendedBilateralFilter(self.num_input_channels, self.dynamic_size, self.fixed_size)



    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]
            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]

            filtered = self.filter(frame_input)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = albedo * filtered

        return output

    def clamp(self):
        self.filter.clamp()

class EncoderUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()

        internal_channels = num_output_channels
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
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()

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

        self.encoder = EncoderUNet2(total_input_channels, num_feature_channels)
        self.filter = SemiFixedBilateralFilter(num_feature_channels + self.num_hidden_channels + 3, self.num_hidden_channels + 3, kernel_size, dilation)


    def forward(self, input):
        B = input.size(0)
        H = input.size(2)
        W = input.size(3)

        # extract features
        features = self.encoder(input)

        combined = torch.cat((input[:, 0:3 + self.num_hidden_channels , :, :], features), dim=1)
        filtered = self.filter(combined)

        return filtered

    def clamp(self):
        self.filter.clamp()

class SimpleFPCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using FPCNN 3")

        self.num_input_channels = 12
        self.num_feature_channels = 8
        self.num_hidden_states = 1
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

        with torch.no_grad():
            self.alpha.abs_()

        self.filters = nn.ModuleList([
            ColorStableBilateralFilter(self.num_input_channels, self.num_hidden_states, self.num_feature_channels, 7, 1)
        ])

        for i in range(4):
            self.filters.append(ColorStableBilateralFilter(self.num_input_channels, self.num_hidden_states, self.num_feature_channels, 5, 2 ** i))

        self.hidden_state_builder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels * 2, self.num_input_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels * 4, self.num_hidden_states, kernel_size=3, padding=1),
        )

        self.clamp()

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = torch.zeros(B, self.num_hidden_states + 3, H, W, device=input.device)

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]
            # albedo is channels 3..5
            albedo = frame_input[:, 3:6, :, :]
            fixed = frame_input[:, 3:, :, :]

            hidden_state = self.hidden_state_builder(frame_input)
            dynamic = torch.cat((frame_input[:, :3, :, :], hidden_state), dim=1)

            if i != 0:
                dynamic = self.alpha * dynamic + (1.0 - self.alpha) * temporal_state

            filtered = torch.cat((dynamic, fixed), dim=1)

            # loop unrolling cuz pytorch is dum dum sometimes
            filtered = self.exec_filter(filtered, fixed, 0)
            filtered = self.exec_filter(filtered, fixed, 1)
            filtered = self.exec_filter(filtered, fixed, 2)
            filtered = self.exec_filter(filtered, fixed, 3)
            filtered = self.exec_filter(filtered, fixed, 4)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = albedo * filtered[:, :3, :, :]

            temporal_state = filtered[:, :3 + self.num_hidden_states, :, :]
        return output

    def clamp(self):
        for i, filter in enumerate(self.filters):
            filter.clamp()

        with torch.no_grad():
            self.alpha.clamp_(0.0, 1.0)

    def exec_filter(self, input, fixed, i):
        filtered = self.filters[i](input)
        combined = torch.cat((filtered, fixed), dim=1)
        return combined
