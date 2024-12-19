import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *
from filter import *
import torch.utils.checkpoint as checkpoint

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
            self.filters.append(ColorStableBilateralFilter(self.num_input_channels, self.num_hidden_states, self.num_feature_channels, 7, 1))

        self.color_recombiner = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(12, 3, kernel_size=3, padding=1, dilation=1),
        )

        self.hidden_state_builder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_hidden_states, kernel_size=3, padding=1),
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
            color = frame_input[:, :3, :, :]
            albedo = frame_input[:, 3:6, :, :]
            fixed = frame_input[:, 3:, :, :]

            color = frame_input[:, :3, :, :]
            norm_color = run_lcn(color, 3)

            recombiner_input = torch.cat((norm_color, color), dim=1)
            recombined = self.color_recombiner(recombiner_input)

            hidden_state_input = torch.cat((recombined, frame_input[:, 3:, :, :]), dim=1)

            hidden_state = self.hidden_state_builder(hidden_state_input)
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









class SimpleFPCNN4(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using FPCNN 4")

        self.num_input_channels = 12
        self.num_feature_channels = 8
        self.num_hidden_states = 1

        num_total_input_channels = self.num_feature_channels + self.num_hidden_states + 3
        self.filters = nn.ModuleList([
            AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder2, 7, 1)
        ])

        for i in range(3):
            self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder2, 5, 1))

        self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder2, 7, 2))
        self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder2, 7, 2))

        base_enc_internal_channels = self.num_input_channels + 3 # extra 3 come from LCN preprocess

        self.base_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, base_enc_internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 4, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
        )

        self.skip_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, base_enc_internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 4, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
        )

        alpha_feature_channels = self.num_feature_channels + self.num_hidden_states + 3
        self.alpha_extractor = nn.Sequential(
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

        self.final_denoiser = nn.Sequential(
            DenoiseUNet(alpha_feature_channels + 3, 1),
        )
        self.clamp_bounds = 1.4

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
            color = frame_input[:, :3, :, :]
            albedo = frame_input[:, 3:6, :, :]

            norm_color = run_lcn(color, 3)

            enc_input = torch.cat((frame_input, norm_color), dim=1)

            init_features = checkpoint.checkpoint_sequential(self.base_encoder, input=enc_input, segments=1)
            skip_features = checkpoint.checkpoint_sequential(self.skip_encoder, input=enc_input, segments=1)

            preprocessed_input = torch.cat((color, init_features), dim=1)

            dynamic = preprocessed_input[:, :3 + self.num_hidden_states, :, :]
            fixed = preprocessed_input[:, 3 + self.num_hidden_states:, :, :]

            if i != 0:
                alpha = checkpoint.checkpoint_sequential(self.alpha_extractor, input=preprocessed_input, segments=1)
                dynamic = alpha[:, 0, :, :].unsqueeze(1) * dynamic + alpha[:, 1, :, :].unsqueeze(1) * temporal_state

            filtered = torch.cat((dynamic, fixed), dim=1)

            # loop unrolling cuz pytorch is dum dum sometimes
            filtered = self.exec_checkpoint_filter(filtered, skip_features, 0)
            filtered = self.exec_checkpoint_filter(filtered, skip_features, 1)
            filtered = self.exec_checkpoint_filter(filtered, skip_features, 2)
            filtered = self.exec_checkpoint_filter(filtered, skip_features, 3)
            filtered = self.exec_checkpoint_filter(filtered, skip_features, 4)
            filtered = self.exec_checkpoint_filter(filtered, skip_features, 5)

            norm_color = run_lcn(filtered[:, :3, :, :], 5)
            filtered = torch.cat((filtered, norm_color), dim=1)

            if False:
                readjustment_factor = self.final_denoiser(filtered)
                readjustment_factor = torch.clamp(readjustment_factor, -self.clamp_bounds, self.clamp_bounds)
                readjustment_factor = torch.exp(readjustment_factor)
                denoised = readjustment_factor * filtered[:, :3, :, :]
            else:
                denoised = self.exec_checkpoint_final_denoise(self.final_denoiser, filtered)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = albedo * denoised[:, :3, :, :]

            temporal_state = denoised#torch.cat((denoised, filtered[:, 3:3 + self.num_hidden_states, :, :]), dim=1)
        return output

    """
    def exec_filter(self, input, skip, i):
        filtered = self.filters[i](input)
        readded_features = filtered[:, 3:, :, :] + skip
        combined = torch.cat((filtered[:, :3, :, :], readded_features), dim=1)
        return combined
    """

    def checkpoint_filter(self, filter):
        def checkpoint_func(*args):
            input = args[0]
            skip = args[1]

            filtered = filter(input)
            readded_features = filtered[:, 3:, :, :] + skip
            combined = torch.cat((filtered[:, :3, :, :], readded_features), dim=1)

            return combined

        return checkpoint_func

    def exec_checkpoint_filter(self, input, skip, i):
        output = checkpoint.checkpoint(self.checkpoint_filter(self.filters[i]), input, skip)
        return output

    def checkpoint_final_denoise(self, module):
        def checkpoint_func(*args):
            input = args[0]
            return module(input)

        return checkpoint_func

    def exec_checkpoint_final_denoise(self, module, input):
        output = checkpoint.checkpoint(self.checkpoint_final_denoise(module), input)
        return output





"""
class SimpleFPCNN5(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using FPCNN 5")

        self.num_input_channels = 12
        self.num_compact_channels = 8
        self.num
        self.num_hidden_states = 1
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

        with torch.no_grad():
            self.alpha.abs_()

        num_total_input_channels = self.num_feature_channels + self.num_hidden_states + 3
        self.filters = nn.ModuleList([
            AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, 7, 1)
        ])

        for i in range(3):
            self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, 5, 1))

        self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, 7, 2))
        self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, 7, 2))

        base_enc_internal_channels = self.num_input_channels + 3 # extra 3 come from LCN preprocess

        self.base_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, base_enc_internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 4, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
        )

        self.skip_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, base_enc_internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 4, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
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
            color = frame_input[:, :3, :, :]
            albedo = frame_input[:, 3:6, :, :]

            color = frame_input[:, :3, :, :]
            norm_color = run_lcn(color, 3)

            enc_input = torch.cat((frame_input, norm_color), dim=1)
            features = self.base_encoder(enc_input)

            init_features = features[:, :self.num_feature_channels + self.num_hidden_states, :, :]
            skip_features = features[:, self.num_feature_channels + self.num_hidden_states:, :, :]

            preprocessed_input = torch.cat((color, init_features), dim=1)

            dynamic = preprocessed_input[:, :3 + self.num_hidden_states, :, :]
            fixed = preprocessed_input[:, 3 + self.num_hidden_states:, :, :]

            if i != 0:
                dynamic = self.alpha * dynamic + (1.0 - self.alpha) * temporal_state

            filtered = torch.cat((dynamic, fixed), dim=1)

            # loop unrolling cuz pytorch is dum dum sometimes
            filtered = self.exec_filter(filtered, skip_features, 0)
            filtered = self.exec_filter(filtered, skip_features, 1)
            filtered = self.exec_filter(filtered, skip_features, 2)
            filtered = self.exec_filter(filtered, skip_features, 3)
            filtered = self.exec_filter(filtered, skip_features, 4)
            filtered = self.exec_filter(filtered, skip_features, 5)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = albedo * filtered[:, :3, :, :]

            temporal_state = filtered[:, :3 + self.num_hidden_states, :, :]
        return output

    def clamp(self):
        for i, filter in enumerate(self.filters):
            filter.clamp()

        with torch.no_grad():
            self.alpha.clamp_(0.0, 1.0)

    def exec_filter(self, input, skip, i):
        filtered = self.filters[i](input)
        readded_features = filtered[:, 3:, :, :] + skip
        combined = torch.cat((filtered[:, :3, :, :], readded_features), dim=1)
        return combined
"""

class EncoderAugmentedFilter(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using Encoder Augmented Filter")

        self.num_input_channels = 12
        self.num_feature_channels = 8
        self.num_hidden_states = 1

        num_total_input_channels = self.num_feature_channels + self.num_hidden_states + 3
        self.filters = nn.ModuleList([
            #AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder3, 7, 1)
        ])

        for i in range(3):
            self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder3, 5, 1))

        for i in range(3):
            self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder3, 5, 2))

        #self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder3, 7, 2))
        #self.filters.append(AuxEncoderBilateralFilter(num_total_input_channels, self.num_hidden_states, self.num_feature_channels, ConvEncoder3, 7, 2))

        base_enc_internal_channels = self.num_input_channels + 3 # extra 3 come from LCN preprocess

        self.base_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, base_enc_internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 4, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
        )

        self.skip_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, base_enc_internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 4, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
        )

        alpha_feature_channels = self.num_feature_channels + self.num_hidden_states + 3
        self.alpha_extractor = nn.Sequential(
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

        self.clamp_bounds = 1.4

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

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def exec_filter(self, input, skip, i):
        filtered = self.filters[i](input)
        readded_features = filtered[:, 3:, :, :] + skip
        combined = torch.cat((filtered[:, :3, :, :], readded_features), dim=1)
        return combined

    def run_frame(self, frame_input, temporal_state, i):
        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        norm_color = run_lcn(color, 3)

        enc_input = torch.cat((frame_input, norm_color), dim=1)

        init_features = self.base_encoder(enc_input)
        skip_features = self.skip_encoder(enc_input)

        preprocessed_input = torch.cat((color, init_features), dim=1)

        dynamic = preprocessed_input[:, :3 + self.num_hidden_states, :, :]
        fixed = preprocessed_input[:, 3 + self.num_hidden_states:, :, :]

        if i != 0:
            alpha = self.alpha_extractor(preprocessed_input)
            dynamic = alpha[:, 0, :, :].unsqueeze(1) * dynamic + alpha[:, 1, :, :].unsqueeze(1) * temporal_state

        filtered = torch.cat((dynamic, fixed), dim=1)

        # loop unrolling cuz pytorch is dum dum sometimes
        filtered = self.exec_filter(filtered, skip_features, 0)
        filtered = self.exec_filter(filtered, skip_features, 1)
        filtered = self.exec_filter(filtered, skip_features, 2)
        filtered = self.exec_filter(filtered, skip_features, 3)
        filtered = self.exec_filter(filtered, skip_features, 4)
        filtered = self.exec_filter(filtered, skip_features, 5)

        norm_color = run_lcn(filtered[:, :3, :, :], 5)
        filtered = torch.cat((filtered, norm_color), dim=1)

        denoised = filtered[:, :3 + self.num_hidden_states, :, :]

        frame_output = albedo * denoised[:, :3, :, :]

        temporal_state = denoised

        return (frame_output, temporal_state)

class EncoderAugmentedFilter2(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using Encoder Augmented Filter2")

        self.num_input_channels = 12
        self.num_passthrough_channels = 11
        self.num_downturn_channels = 8
        self.num_hidden_states = 1

        self.filters = nn.ModuleList([])

        for i in range(4):
            self.filters.append(DownturnEncoderBilateralFilter(self.num_passthrough_channels, self.num_hidden_states, self.num_downturn_channels, ConvEncoder4, ConvEncoder5, 5, 1))

        for i in range(4):
            self.filters.append(DownturnEncoderBilateralFilter(self.num_passthrough_channels, self.num_hidden_states, self.num_downturn_channels, ConvEncoder4, ConvEncoder5, 5, 2))

        base_enc_internal_channels = self.num_input_channels + 3 # extra 3 come from LCN preprocess
        self.base_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, base_enc_internal_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 4, self.num_passthrough_channels + self.num_hidden_states, kernel_size=3, padding=1),
        )

        alpha_feature_channels = self.num_passthrough_channels + self.num_hidden_states + 3
        self.alpha_extractor = nn.Sequential(
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

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

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def exec_filter(self, input, i):
        skip = input[:, 3:, :, :]
        filtered = self.filters[i](input)
        readded_features = filtered[:, 3:, :, :] + skip
        combined = torch.cat((filtered[:, :3, :, :], readded_features), dim=1)
        return combined

    def run_frame(self, frame_input, temporal_state, i):
        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        norm_color = run_lcn(color, 3)

        enc_input = torch.cat((frame_input, norm_color), dim=1)

        init_features = self.base_encoder(enc_input)

        preprocessed_input = torch.cat((color, init_features), dim=1)

        dynamic = preprocessed_input[:, :3 + self.num_hidden_states, :, :]
        fixed = preprocessed_input[:, 3 + self.num_hidden_states:, :, :]

        if i != 0:
            alpha = self.alpha_extractor(preprocessed_input)
            dynamic = alpha[:, 0, :, :].unsqueeze(1) * dynamic + alpha[:, 1, :, :].unsqueeze(1) * temporal_state

        filtered = torch.cat((dynamic, fixed), dim=1)

        # loop unrolling cuz pytorch is dum dum sometimes
        filtered = self.exec_filter(filtered, 0)
        filtered = self.exec_filter(filtered, 1)
        filtered = self.exec_filter(filtered, 2)
        filtered = self.exec_filter(filtered, 3)
        filtered = self.exec_filter(filtered, 4)
        filtered = self.exec_filter(filtered, 5)
        filtered = self.exec_filter(filtered, 6)
        filtered = self.exec_filter(filtered, 7)

        norm_color = run_lcn(filtered[:, :3, :, :], 5)
        filtered = torch.cat((filtered, norm_color), dim=1)

        denoised = filtered[:, :3 + self.num_hidden_states, :, :]

        frame_output = albedo * denoised[:, :3, :, :]

        temporal_state = denoised

        return (frame_output, temporal_state)

class UNetDLF(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using UNDLF")

        self.num_input_channels = 12
        self.num_feature_channels = 13
        self.num_hidden_states = 3


        num_total_input_channels = self.num_feature_channels + self.num_hidden_states + 3

        self.filters = nn.ModuleList([])
        for i in range(4):
            self.filters.append(WeightTransformBilateralFilter(num_total_input_channels, self.num_hidden_states + 3, 5, 1))

        base_enc_internal_channels = self.num_input_channels + 3 # extra 3 come from LCN preprocess
        self.base_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
        )

        alpha_feature_channels = self.num_feature_channels + self.num_hidden_states + 3
        self.alpha_extractor = nn.Sequential(
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

        self.encoders = nn.ModuleList([])
        for i in range(4):
            self.encoders.append(
                EESPEncoderSimple(self.num_feature_channels + self.num_hidden_states, self.num_hidden_states, 4, 1)
            )

        num_digest_channels_per_layer = 3

        self.digest_encoders = nn.ModuleList([])
        for i in range(4):
            self.digest_encoders.append(
                nn.Sequential(
                    EESPEncoderSimple(self.num_feature_channels + self.num_hidden_states, self.num_hidden_states, 4, 1),
                    nn.ReLU(),
                    nn.Conv2d(self.num_feature_channels + self.num_hidden_states, self.num_feature_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.num_feature_channels, num_digest_channels_per_layer, kernel_size=3, padding=1),
                )
            )

        self.upscale = nn.ModuleList([])
        for i in range(3):
            scale_factor = 2 ** (i + 1)
            self.upscale.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True))

        self.fixed_pool = nn.ModuleList([])
        for i in range(3):
            self.fixed_pool.append(
                nn.Sequential(
                    nn.Conv2d(self.num_feature_channels, self.num_feature_channels, kernel_size=1, groups=self.num_feature_channels),
                    nn.MaxPool2d(2)
                )
            )

        self.dyn_pool = nn.ModuleList([])
        for i in range(3):
            self.dyn_pool.append(nn.AvgPool2d(2))

        weighting_channels = 4 * num_digest_channels_per_layer
        self.weight_net = nn.Sequential(
            nn.Conv2d(weighting_channels, weighting_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(weighting_channels, weighting_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(weighting_channels, weighting_channels * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(weighting_channels * 2, weighting_channels * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(weighting_channels * 4, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )

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

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        norm_color = run_lcn(color, 3)

        enc_input = torch.cat((frame_input, norm_color), dim=1)

        init_features = self.base_encoder(enc_input)

        preprocessed_input = torch.cat((color, init_features), dim=1)

        dynamic = preprocessed_input[:, :3 + self.num_hidden_states, :, :]
        fixed = preprocessed_input[:, 3 + self.num_hidden_states:, :, :]

        if i != 0:
            alpha = self.alpha_extractor(preprocessed_input)
            dynamic = alpha[:, 0, :, :].unsqueeze(1) * dynamic + alpha[:, 1, :, :].unsqueeze(1) * temporal_state

        filtered = torch.cat((dynamic, fixed), dim=1)

        filtered_dynamic = []
        digested_info = []
        for i in range(4):
            encoding = self.encoders[i](filtered)

            filtered = torch.cat((filtered[:, :3, :, :], encoding), dim=1)

            filtered = self.filters[i](filtered)

            skip_hidden = filtered[:, 3:, :, :] + encoding[:, :self.num_hidden_states, :, :]
            filtered = torch.cat((filtered[:, :3, :, :], skip_hidden, encoding[:, self.num_hidden_states:, :, :]), dim=1)


            cur_dyn = filtered[:, :3 + self.num_hidden_states, :, :]
            if i == 0:
                digested_info.append(self.digest_encoders[i](filtered))
                filtered_dynamic.append(cur_dyn)
            else:
                upscale = self.upscale[i - 1]
                digested_info.append(self.digest_encoders[i](upscale(filtered)))
                filtered_dynamic.append(upscale(cur_dyn))


            if i != 3:
                dyn_ds = self.dyn_pool[i](filtered[:, :3 + self.num_hidden_states, :, :])
                fixed_ds = self.fixed_pool[i](filtered[:, 3 + self.num_hidden_states:, :, :])

                filtered = torch.cat((dyn_ds, fixed_ds), dim=1)

        weights = self.weight_net(torch.cat(digested_info, dim=1))

        for i in range(4):
            weighted = weights[:, i, :, :].unsqueeze(1) * filtered_dynamic[i]
            if i == 0:
                denoised = weighted
            else:
                denoised = denoised + weighted

        frame_output = albedo * denoised[:, :3, :, :]

        temporal_state = denoised

        return (frame_output, temporal_state)

# this is designed to be the heavyweight reference
class UNetDLF2(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using UNDLF2")

        self.num_input_channels = 12
        self.num_feature_channels = 8
        self.num_hidden_states = 3

        base_enc_internal_channels = self.num_input_channels + 3 # extra 3 come from LCN preprocess
        self.base_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
        )

        alpha_feature_channels = self.num_feature_channels + self.num_hidden_states + 3
        self.alpha_extractor = nn.Sequential(
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, alpha_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(alpha_feature_channels, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

        self.encoders = nn.ModuleList([])
        self.filters = nn.ModuleList([])
        self.pool = nn.ModuleList([])
        self.upscale = nn.ModuleList([])

        num_extracted_channels = self.num_feature_channels + self.num_hidden_states
        total_weight_inputs = 0
        for i in range(4):
            input_size = num_extracted_channels * (2 ** i)

            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_size + 3, input_size * 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(input_size * 2, input_size * 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(input_size * 2, input_size * 2, kernel_size=3, padding=1)
                )
            )

            total_weight_inputs += input_size * 2 + 3
            self.filters.append(WeightTransformBilateralFilter(input_size * 2 + 3, self.num_hidden_states + 3, 5, 1))

            if i < 3:
                self.pool.append(ColorStabilizingPool(3))

            if i > 0:
                self.upscale.append(nn.ConvTranspose2d(input_size * 2 + 3, input_size * 2 + 3, kernel_size=(2 ** i), stride=(2 ** i), groups=input_size * 2 + 3))

        self.direct_predictor = nn.Sequential(
            nn.Conv2d(total_weight_inputs, total_weight_inputs, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(total_weight_inputs, total_weight_inputs, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(total_weight_inputs, total_weight_inputs, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(total_weight_inputs, 3 + self.num_hidden_states, kernel_size=3, padding=1),
        )

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

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        norm_color = run_lcn(color, 3)

        enc_input = torch.cat((frame_input, norm_color), dim=1)

        init_features = self.base_encoder(enc_input)

        preprocessed_input = torch.cat((color, init_features), dim=1)

        dynamic = preprocessed_input[:, :3 + self.num_hidden_states, :, :]
        fixed = preprocessed_input[:, 3 + self.num_hidden_states:, :, :]

        if i != 0:
            alpha = self.alpha_extractor(preprocessed_input)
            dynamic = alpha[:, 0, :, :].unsqueeze(1) * dynamic + alpha[:, 1, :, :].unsqueeze(1) * temporal_state

        filtered = torch.cat((dynamic, fixed), dim=1)

        skip_context = []
        for i in range(4):
            encoding = self.encoders[i](filtered)

            filtered = torch.cat((filtered[:, :3, :, :], encoding), dim=1)
            filtered = self.filters[i](filtered)
            filtered = torch.cat((filtered[:, :3, :, :], encoding), dim=1)

            upsampled = filtered
            if i > 0:
                upsampled = self.upscale[i - 1](upsampled)

            skip_context.append(upsampled)

            if i < 3:
                filtered = self.pool[i](filtered)


        denoised = self.direct_predictor(torch.cat(skip_context, dim=1))

        frame_output = albedo * denoised[:, :3, :, :]

        temporal_state = denoised

        return (frame_output, temporal_state)
