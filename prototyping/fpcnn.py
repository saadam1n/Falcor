import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *
from filter import *
import torch.utils.checkpoint as checkpoint
import keyboard, cv2
import torchvision.transforms.functional as F
import math

def display_intermediate(tensor, key):
    if keyboard.is_pressed(key):
        image = torch.clone(tensor).detach().pow(1 / 2.2).squeeze().permute((1, 2, 0)).cpu().numpy()
        first_iter = True
        while keyboard.is_pressed(key):
            if first_iter:
                print("DISPLAY INTERMEDIATE. Please release the key to display the window.")
                first_iter = False


        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

class UNetDLFNonEesp(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using UNDLF 1.0 (EESP Free Edition)")

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
                nn.Sequential(
                    nn.Conv2d(self.num_feature_channels + self.num_hidden_states + 3, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.num_feature_channels + self.num_hidden_states, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.num_feature_channels + self.num_hidden_states, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),
                )

            )

        num_digest_channels_per_layer = 3

        self.digest_encoders = nn.ModuleList([])
        for i in range(4):
            self.digest_encoders.append(
                nn.Sequential(
                    nn.Conv2d(self.num_feature_channels + self.num_hidden_states + 3, self.num_feature_channels + self.num_hidden_states, kernel_size=3, padding=1),

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
        self.decoders = nn.ModuleList([])

        num_extracted_channels = self.num_feature_channels + self.num_hidden_states
        for i in range(4):
            input_size = num_extracted_channels * (2 ** i)

            output_size = input_size * 2 if i != 3 else input_size
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_size + 3, input_size * 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(input_size * 2, input_size * 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(input_size * 2, output_size, kernel_size=3, padding=1)
                )
            )

            self.filters.append(WeightTransformBilateralFilter(output_size + 3, self.num_hidden_states + 3, 5, 1))

            if i < 3:
                self.pool.append(ColorStabilizingPool(3))

            if i > 0:
                upscale_input_size = input_size if i != 3 else input_size + 3
                self.upscale.append(nn.ConvTranspose2d(upscale_input_size, input_size, kernel_size=2, stride=2))

            if i < 3:
                print(f"At {i} our input size is {input_size}")

                decoder_output_size = input_size if i != 0 else 3 + self.num_hidden_states
                decoder = nn.Sequential(
                    nn.Conv2d(4 * input_size + 3, 2 * input_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(2 * input_size, 2 * input_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(2 * input_size, decoder_output_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                )

                if False:
                    for layer in decoder:
                        if isinstance(layer, nn.Conv2d):
                            with torch.no_grad():
                                layer.weight[:, :3, :, :].mul_(4.0)
                                layer.weight[:, 3:, :, :].mul_(0.0)

                self.decoders.append(
                    decoder
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
        # (0 1) (1 2) (2 3) (3 3)
        for i in range(4):
            encoding = self.encoders[i](filtered)

            filtered = torch.cat((filtered[:, :3, :, :], encoding), dim=1)
            filtered = self.filters[i](filtered)
            filtered = torch.cat((filtered[:, :3, :, :], encoding), dim=1)

            if i < 3:
                skip_context.append(filtered)
                filtered = self.pool[i](filtered)

        # (3 2) (2, 1) (1 0)
        for i in range(3):
            j = 3 - i - 1

            filtered = self.upscale[j](filtered)

            filtered = torch.cat((filtered, skip_context[j]), dim=1)

            filtered = self.decoders[j](filtered)

        denoised = filtered

        frame_output = albedo * denoised[:, :3, :, :]

        temporal_state = denoised

        return (frame_output, temporal_state)


class UNetDLF3(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using UNDLF3")

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
        self.unfold_ds = nn.ModuleList([])
        self.unfold_us = nn.ModuleList([])
        self.unfold_fr = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        num_extracted_channels = self.num_feature_channels + self.num_hidden_states
        for i in range(4):
            input_size = num_extracted_channels * (2 ** i)

            output_size = input_size * 2 if i != 3 else input_size
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_size + 3, input_size * 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(input_size * 2, input_size * 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(input_size * 2, output_size, kernel_size=3, padding=1)
                )
            )

            self.filters.append(WeightTransformBilateralFilter(output_size + 3, self.num_hidden_states + 3, 5, 1))

            if i < 3:
                self.pool.append(ColorStabilizingPool(3))

            if i > 0:
                upscale_input_size = input_size if i != 3 else input_size + 3
                self.upscale.append(nn.ConvTranspose2d(upscale_input_size, input_size, kernel_size=2, stride=2))

            if i < 3:
                print(f"At {i} our input size is {input_size}")

                self.unfold_ds.append(nn.Unfold(kernel_size=3, padding=1))
                self.unfold_fr.append(nn.Unfold(kernel_size=3, padding=1))
                self.unfold_us.append(nn.Upsample(scale_factor=2, mode="nearest"))

                decoder_output_size = input_size - 3 + 18 if i != 0 else 18 + self.num_hidden_states
                decoder = nn.Sequential(
                    nn.Conv2d(4 * input_size + 3, 2 * input_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(2 * input_size, 2 * input_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(2 * input_size, decoder_output_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                )

                self.decoders.append(
                    decoder
                )

        self.artifact_remover = nn.Sequential(
            nn.Conv2d(3 + self.num_hidden_states, 6 + 2 * self.num_hidden_states, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(6 + 2 * self.num_hidden_states, 6 + 2 * self.num_hidden_states, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(6 + 2 * self.num_hidden_states, 3 + self.num_hidden_states, kernel_size=3, padding=1),
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
        # (0 1) (1 2) (2 3) (3 3)
        for i in range(4):
            encoding = self.encoders[i](filtered)

            filtered = torch.cat((filtered[:, :3, :, :], encoding), dim=1)
            filtered = self.filters[i](filtered)
            filtered = torch.cat((filtered[:, :3, :, :], encoding), dim=1)

            if i < 3:
                skip_context.append(filtered)
                filtered = self.pool[i](filtered)

        # (3 2) (2, 1) (1 0)
        for i in range(3):
            j = 3 - i - 1

            ds_window = self.unfold_ds[j](filtered[:, :3, :, :])
            fr_window = self.unfold_fr[j](skip_context[j][:, :3, :, :])

            ds_window = ds_window.view(filtered.size(0), -1, filtered.size(2), filtered.size(3))
            ds_window = self.unfold_us[j](ds_window)

            fr_window = fr_window.view(ds_window.size(0), -1, ds_window.size(2), ds_window.size(3))

            window = torch.cat((fr_window, ds_window), dim=1)

            window = window.view(window.size(0), 18, 3, window.size(2), window.size(3))


            filtered = self.upscale[j](filtered)
            filtered = torch.cat((filtered, skip_context[j]), dim=1)
            filtered = self.decoders[j](filtered)

            weights = F.softmax(filtered[:, :18, :, :], dim=1)
            weights = weights.unsqueeze(2)

            mult = weights * window
            accumulated = mult.sum(1)

            filtered = torch.cat((accumulated, filtered[:, 18:, :, :]), dim=1)


        denoised = self.artifact_remover(filtered)

        frame_output = albedo * denoised[:, :3, :, :]

        temporal_state = denoised

        return (frame_output, temporal_state)



class UNetDLF4(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using UNDLF4")

        self.num_input_channels = 12
        self.num_total_channels = 16
        self.num_dynamic_states = 6

        base_enc_internal_channels = self.num_input_channels + 3 # extra 3 come from LCN preprocess
        self.base_encoder = nn.Sequential(
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels, base_enc_internal_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(base_enc_internal_channels * 2, self.num_total_channels, kernel_size=5, padding=2),
        )

        self.alpha_extractor = nn.Sequential(
            nn.Conv2d(self.num_total_channels + self.num_dynamic_states, self.num_total_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_total_channels, self.num_total_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_total_channels, 2, kernel_size=5, padding=2),
            nn.Softmax(dim=1)
        )

        self.encoders = nn.ModuleList([])
        self.filters = nn.ModuleList([])
        self.pool = nn.ModuleList([])
        self.upscale = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        # a + b
        # 2(a + b)
        # first a channels remained fixed, so we create 2(a + b) - a channels
        # a + 2b channels are created
        # the question becomes how do we handle i=0?
        # a and b are inputs *at that level*

        for i in range(4):
            input_size = self.num_total_channels * (2 ** i)
            dynamic_input_size = self.num_dynamic_states * (2 ** i)
            fixed_input_size = input_size - dynamic_input_size

            output_size = input_size + fixed_input_size if i != 3 else input_size
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_size, input_size * 2, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(input_size * 2, input_size * 2, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(input_size * 2, output_size, kernel_size=5, padding=2)
                )
            )

            print(f"Input channels at {i} are {output_size + dynamic_input_size}")
            print(f"Decoder out is {output_size}")
            print(f"Output should be {dynamic_input_size * 2}")
            self.filters.append(WeightTransformBilateralFilter(output_size + dynamic_input_size, dynamic_input_size * 2, 5, 1))

            if i < 3:
                self.pool.append(ColorStabilizingPool(self.num_dynamic_states))

            if i > 0:
                upscale_input_size = output_size + dynamic_input_size if i == 3 else input_size
                self.upscale.append(nn.ConvTranspose2d(upscale_input_size, input_size, kernel_size=2, stride=2))

            if i < 3:
                decoder_output_size = self.num_dynamic_states + 3 if i == 0 else input_size
                decoder = nn.Sequential(
                    nn.Conv2d(4 * input_size, 4 * input_size, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(4 * input_size, 2 * input_size, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(2 * input_size, decoder_output_size, kernel_size=5, padding=2),
                    nn.ReLU(),
                )

                self.decoders.append(
                    decoder
                )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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

        dynamic = init_features[:, :self.num_dynamic_states, :, :]
        fixed = init_features[:, self.num_dynamic_states:, :, :]

        if i != 0:
            alpha = self.alpha_extractor(torch.cat((init_features, temporal_state), dim=1))
            dynamic = alpha[:, 0, :, :].unsqueeze(1) * dynamic + alpha[:, 1, :, :].unsqueeze(1) * temporal_state

        filtered = torch.cat((dynamic, fixed), dim=1)

        skip_context = []
        # (0 1) (1 2) (2 3) (3 3)
        for i in range(4):

            encoding = self.encoders[i](filtered)
            prev_dyn = self.num_dynamic_states * (2 ** i)
            cur_skip = filtered[:, :prev_dyn, :, :]
            filtered = torch.cat((cur_skip, encoding), dim=1)
            filtered = self.filters[i](filtered)
            filtered = torch.cat((filtered[:, :prev_dyn, :, :] + cur_skip, filtered[:, prev_dyn:, :, :], encoding[:, prev_dyn:, :, :]), dim=1)



            if i < 3:
                skip_context.append(filtered)
                filtered = self.pool[i](filtered)

        # (3 2) (2, 1) (1 0)
        for i in range(3):
            j = 3 - i - 1

            filtered = self.upscale[j](filtered)

            filtered = torch.cat((filtered, skip_context[j]), dim=1)

            filtered = self.decoders[j](filtered)


        denoised = filtered

        frame_output = albedo * denoised[:, :3, :, :]

        temporal_state = denoised[:, 3:, :, :]

        return (frame_output, temporal_state)

class WeightPredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using Weight Prediction Network 1.0")


        self.num_input_channels = 12
        self.num_features_1x1 = 4
        self.num_features_3x3 = 6
        self.num_features_5x5 = 8
        self.num_features_tot = self.num_features_1x1 + self.num_features_3x3 + self.num_features_5x5
        self.num_dynamic_channels = 3

        self.feat_extractor_1x1 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_1x1, kernel_size=1, padding=0)
        )

        self.feat_extractor_3x3 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_3x3, kernel_size=3, padding=1)
        )

        self.feat_extractor_5x5 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_5x5, kernel_size=5, padding=2)
        )

        self.alpha_calc = nn.Sequential(
            nn.Conv2d(self.num_input_channels + self.num_dynamic_channels + 3, self.num_input_channels + self.num_dynamic_channels + 3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + self.num_dynamic_channels + 3, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.num_filters = 3
        dilations = [1, 1, 1, 2, 2, 4]
        self.feature_updater = nn.ModuleList([])
        self.filters = nn.ModuleList([])
        for i in range(self.num_filters):
            self.feature_updater.append(
                nn.Sequential(
                    nn.Conv2d(self.num_features_tot + 3, self.num_features_tot + 3, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.num_features_tot + 3, self.num_features_tot * 2 + 5, kernel_size=1, padding=0),
                )
            )


            dilation = dilations[i]
            self.filters.append(
                WeightTransformPixelBilateralFilter(3, 7, dilation)
            )

        self.dynamic_extractor = nn.Sequential(
            nn.Conv2d(self.num_features_tot + 3, self.num_features_tot + 3, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_features_tot + 3, self.num_dynamic_channels, kernel_size=1, padding=0)
        )

        self.max_weight = 10

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        norm_block = run_lcn(frame_input[:, :3, :, :], 7)
        norm_block = torch.cat((norm_block, frame_input[:, 3:, :, :]), dim=1)

        # first lets reproject
        if i != 0:
            alpha = self.alpha_calc(torch.cat((frame_input, temporal_state), dim=1))
            color = alpha[:, :1, :, :] * frame_input[:, :3, :, :] + alpha[:, 1:, :, :] * temporal_state[:, :3, :, :]
        else:
            color = frame_input[:, :3, :, :]

        features_1x1 = self.feat_extractor_1x1(norm_block)
        features_3x3 = self.feat_extractor_3x3(norm_block)
        features_5x5 = self.feat_extractor_5x5(norm_block)
        embedding = torch.cat((color, features_1x1, features_3x3, features_5x5), dim=1)

        for i in range(self.num_filters):
            skip = embedding[:, 3:, :, :]

            norm_block = run_lcn(embedding[:, :3, :, :], 7)
            norm_block = torch.cat((norm_block, skip), dim=1)
            update = self.feature_updater[i](norm_block)

            features = update[:, :self.num_features_tot, :, :]
            params = update[:, self.num_features_tot:, :, :]

            params = -params * params

            color = embedding[:, :3, :, :]
            embedding = torch.cat((color, features), dim=1)

            color = self.filters[i](embedding, params)

            embedding = torch.cat((color, features + skip), dim=1)

        denoised = embedding[:, :3, :, :]

        temporal_state = self.dynamic_extractor(embedding)
        temporal_state = torch.cat((denoised, temporal_state), dim=1)

        frame_output = albedo * denoised

        return (frame_output, temporal_state)


class WeightPredictionNetwork2_LQ(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using Weight Prediction Network 2.0 (LQ)")


        self.num_input_channels = 12
        self.num_features_1x1 = 4
        self.num_features_3x3 = 4
        self.num_features_5x5 = 6
        self.num_features_tot = self.num_features_1x1 + self.num_features_3x3 + self.num_features_5x5
        self.num_dynamic_channels = 3

        self.feat_extractor_1x1_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_1x1, kernel_size=1, padding=0)
        )

        self.feat_extractor_3x3_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_3x3, kernel_size=3, padding=1)
        )

        self.feat_extractor_5x5_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_5x5, kernel_size=5, padding=2)
        )

        self.feat_extractor_1x1_r = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_1x1, kernel_size=1, padding=0)
        )

        self.feat_extractor_3x3_r = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_3x3, kernel_size=3, padding=1)
        )

        self.feat_extractor_5x5_r = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_5x5, kernel_size=5, padding=2)
        )

        self.alpha_calc = nn.Sequential(
            nn.Conv2d(self.num_input_channels + self.num_dynamic_channels + 5, self.num_input_channels + self.num_dynamic_channels + 5, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + self.num_dynamic_channels + 5, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.num_filters = 3
        self.feature_updater = nn.ModuleList([])
        self.filters = nn.ModuleList([])
        for i in range(self.num_filters):
            self.feature_updater.append(
                nn.Sequential(
                    nn.Conv2d(self.num_features_tot + 3, self.num_features_tot + 3, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.num_features_tot + 3, self.num_features_tot * 2 + 5, kernel_size=1, padding=0),
                )
            )

            self.filters.append(
                WeightTransformPixelBilateralFilter(3, 5, 1)
            )

        self.dynamic_extractor = nn.Sequential(
            nn.Conv2d(self.num_features_tot + 3, self.num_features_tot + 3, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_features_tot + 3, self.num_dynamic_channels, kernel_size=1, padding=0)
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        if i != 0:
            per_pixel_info = torch.cat((frame_input, temporal_state), dim=1)

            per_pixel_info2 = per_pixel_info * per_pixel_info

            mean = torch.mean(per_pixel_info, dim=1).unsqueeze(1)
            var = torch.mean(per_pixel_info2, dim=1).unsqueeze(1) - mean * mean
            std = torch.sqrt(var + 1e-6)

            norm_per_pixel_info = (per_pixel_info - mean) / std
            norm_per_pixel_info = torch.cat((norm_per_pixel_info, mean, var), dim=1)

            alpha = self.alpha_calc(norm_per_pixel_info)

            reproj_color = frame_input[:, :3, :, :] * alpha[:, :1, :, :] + temporal_state[:, :3, :, :] * alpha[:, 1:, :, :]

            frame_input = torch.cat((reproj_color, frame_input[:, 3:, :, :]), dim=1)

        # run positional norm
        frame_input2 = frame_input * frame_input
        mean = torch.mean(frame_input, dim=1).unsqueeze(1)
        var = torch.mean(frame_input2, dim=1).unsqueeze(1) - mean * mean
        std = torch.sqrt(var + 1e-6)

        norm_frame_input = (frame_input - mean) / std
        norm_frame_input = torch.cat((norm_frame_input, mean, var), dim=1)

        if i == 0:
            # no reprojection information, use different network
            features_1x1 = self.feat_extractor_1x1_n(norm_frame_input)
            features_3x3 = self.feat_extractor_3x3_n(norm_frame_input)
            features_5x5 = self.feat_extractor_5x5_n(norm_frame_input)
        else:
            # we have reproj information, use different network
            features_1x1 = self.feat_extractor_1x1_r(norm_frame_input)
            features_3x3 = self.feat_extractor_3x3_r(norm_frame_input)
            features_5x5 = self.feat_extractor_5x5_r(norm_frame_input)

        embedding = torch.cat((norm_frame_input[:, :3, :, :], features_1x1, features_3x3, features_5x5), dim=1)

        for i in range(self.num_filters):
            skip = embedding[:, 3:, :, :]

            update = self.feature_updater[i](embedding)

            features = update[:, :self.num_features_tot, :, :]
            params = update[:, self.num_features_tot:, :, :]

            params = -params * params

            color = embedding[:, :3, :, :]
            embedding = torch.cat((color, features), dim=1)

            color = self.filters[i](embedding, params)

            embedding = torch.cat((color, features + skip), dim=1)


        rgb = embedding[:, :3, :, :] * std + mean


        temporal_state = self.dynamic_extractor(embedding)
        temporal_state = torch.cat((rgb, temporal_state), dim=1)

        frame_output = albedo * rgb


        return (frame_output, temporal_state)

class WeightPredictionNetwork3_HQ(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using Weight Prediction Network 3.0 (HQ)")


        self.num_input_channels = 12
        self.num_features_1x1 = 10
        self.num_features_3x3 = 10
        self.num_features_5x5 = 10
        self.num_features_tot = self.num_features_1x1 + self.num_features_3x3 + self.num_features_5x5
        self.num_dynamic_channels = 6
        self.num_embedding_channels = 26
        self.num_expanded_channels = 2 * self.num_embedding_channels - 2 # minus two because we don't need other channels

        self.feat_extractor_1x1_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_1x1, kernel_size=1, padding=0)
        )

        self.feat_extractor_3x3_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_3x3, kernel_size=3, padding=1)
        )

        self.feat_extractor_5x5_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels + 2, self.num_input_channels + 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 2, self.num_features_5x5, kernel_size=5, padding=2)
        )


        self.embedding_builder = nn.Sequential(
            nn.Conv2d(self.num_features_tot, self.num_embedding_channels, kernel_size=1)
        )

        self.filter0_0 = BetterPixelBilateralFilter(self.num_dynamic_channels, 5, 1)
        self.filter0_1 = BetterPixelBilateralFilter(self.num_dynamic_channels, 5, 1)

        self.channel_expander = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels + self.num_features_tot, self.num_dynamic_channels + self.num_features_tot, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels + self.num_features_tot, self.num_expanded_channels, kernel_size=3, padding=1),
        )

        self.pool = nn.Conv2d(self.num_expanded_channels, self.num_expanded_channels, kernel_size=2, stride=2, groups=self.num_expanded_channels)

        self.global_feature_extractor = nn.Sequential(
            nn.Conv2d(self.num_expanded_channels, self.num_expanded_channels, kernel_size=5, padding=2),
        )

        self.filter1_0 = BetterPixelBilateralFilter(self.num_dynamic_channels * 2, 5, 1)
        self.filter1_1 = BetterPixelBilateralFilter(self.num_dynamic_channels * 2, 5, 1)

        self.upscale = nn.ConvTranspose2d(self.num_expanded_channels, self.num_expanded_channels, kernel_size=2, stride=2)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_expanded_channels, self.num_expanded_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_expanded_channels, self.num_expanded_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_expanded_channels, 3, kernel_size=1, padding=0),
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        # run positional norm
        frame_input2 = frame_input * frame_input
        mean = torch.mean(frame_input, dim=1).unsqueeze(1)
        var = torch.mean(frame_input2, dim=1).unsqueeze(1) - mean * mean
        std = torch.sqrt(var + 1e-6)

        norm_frame_input = (frame_input - mean) / std
        norm_frame_input = torch.cat((norm_frame_input, mean, var), dim=1)

        features_1x1 = self.feat_extractor_1x1_n(norm_frame_input)
        features_3x3 = self.feat_extractor_3x3_n(norm_frame_input)
        features_5x5 = self.feat_extractor_5x5_n(norm_frame_input)

        embedding0 = self.embedding_builder(torch.cat((features_1x1, features_3x3, features_5x5), dim=1))


        filtered0 = self.filter0_0(embedding0)
        filtered0 = torch.cat((filtered0, embedding0[:, self.num_dynamic_channels:, :, :]), dim=1)
        filtered0 = self.filter0_1(filtered0)
        filtered0 = torch.cat((filtered0 + embedding0[:, :self.num_dynamic_channels, :, :], embedding0[:, self.num_dynamic_channels:, :, :]), dim=1)

        display_intermediate(filtered0[:, :3, :, :], "f9")
        display_intermediate(filtered0[:, 3:6, :, :], "f8")


        skip = self.channel_expander(torch.cat((filtered0[:, :self.num_dynamic_channels, :, :], features_1x1, features_3x3, features_5x5), dim=1))
        embedding1 = self.pool(skip)

        embedding1 = self.global_feature_extractor(embedding1)

        filtered1 = self.filter1_0(embedding1)
        filtered1 = torch.cat((filtered1, embedding1[:, self.num_dynamic_channels * 2:, :, :]), dim=1)
        filtered1 = self.filter1_1(filtered1)
        filtered1 = torch.cat((filtered1 + embedding1[:, :self.num_dynamic_channels * 2, :, :], embedding1[:, self.num_dynamic_channels * 2:, :, :]), dim=1)

        filtered1 = self.upscale(filtered1) + skip
        display_intermediate(filtered1[:, :3, :, :], "f7")
        display_intermediate(filtered1[:, 3:6, :, :], "f6")


        rgb = self.decoder(filtered1)

        display_intermediate(rgb, "f5")


        frame_output = albedo * rgb


        return (frame_output, temporal_state)


class DifferentialFilterLane(nn.Module):
    def __init__(self, num_feature_channels, num_dynamic_channels):
        super().__init__()

        print("\tUsing a differential filter lane with above network.")

        self.num_feature_channels = num_feature_channels
        self.num_dynamic_channels = num_dynamic_channels

        self.filter = WeightTransformBilateralFilter(self.num_feature_channels, self.num_dynamic_channels, 5, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_feature_channels, self.num_feature_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.num_feature_channels, self.num_feature_channels, kernel_size=1),
        )

    def forward(self, input):
        B = input.size(0)
        H = input.size(2)
        W = input.size(3)

        enc = self.encoder(input)
        filtered = self.filter(enc)

        return filtered


class DifferentialFilter(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using Differential Filter 1.0")


        self.num_input_channels = 12
        self.num_features_1x1 = 8
        self.num_features_3x3 = 8
        self.num_features_5x5 = 8
        self.num_features_tot = self.num_features_1x1 + self.num_features_3x3 + self.num_features_5x5
        self.num_dynamic_channels = 8

        self.feat_extractor_1x1_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_1x1, kernel_size=1, padding=0)
        )

        self.feat_extractor_3x3_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_3x3, kernel_size=3, padding=1)
        )

        self.feat_extractor_5x5_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_5x5, kernel_size=5, padding=2)
        )

        self.filter0 = DifferentialFilterLane(self.num_features_tot, self.num_dynamic_channels)
        self.filter1 = DifferentialFilterLane(self.num_features_tot, self.num_dynamic_channels)
        self.filter2 = nn.Sequential(
            nn.Conv2d(self.num_features_tot, self.num_features_tot, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_features_tot, self.num_features_tot, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_features_tot, self.num_dynamic_channels, kernel_size=3, padding=1),
        )

        self.unet = NoisePredictingUNet(self.num_features_tot + self.num_dynamic_channels * 2 + 4, self.num_dynamic_channels)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_features_tot + self.num_dynamic_channels, self.num_features_tot + self.num_dynamic_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_features_tot + self.num_dynamic_channels, self.num_features_tot + self.num_dynamic_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_features_tot + self.num_dynamic_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        features_1x1 = self.feat_extractor_1x1_n(frame_input)
        features_3x3 = self.feat_extractor_3x3_n(frame_input)
        features_5x5 = self.feat_extractor_5x5_n(frame_input)

        embedding = torch.cat((features_1x1, features_3x3, features_5x5), dim=1)

        filtered0 = self.filter0(embedding)
        filtered1 = self.filter1(embedding)

        diff = filtered0 - filtered1

        # assist noise calculations using estimates of mean and variance
        meand = diff.mean(1).unsqueeze(1)
        vard = (diff * diff).sum(1).unsqueeze(1) - meand * meand

        mean2 = filtered0.mean(1).unsqueeze(1)
        var2 = (filtered0 * filtered0).mean(1).unsqueeze(1) - mean2 * mean2

        compressable = torch.cat((filtered0, diff, embedding, meand, vard, mean2, var2), dim=1)

        noise = self.unet(compressable)

        denoised = filtered0 - noise

        color = self.decoder(torch.cat((denoised, embedding), dim=1))

        frame_output = albedo * color

        return (frame_output, temporal_state)


class PlainUNetFilter(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using PlainUNetFilter")

        self.num_input_channels = 12
        self.num_features_1x1 = 12
        self.num_features_3x3 = 12
        self.num_features_5x5 = 12
        self.num_features_tot = self.num_features_1x1 + self.num_features_3x3 + self.num_features_5x5

        self.feat_extractor_1x1_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_1x1, kernel_size=1, padding=0)
        )

        self.feat_extractor_3x3_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_3x3, kernel_size=3, padding=1)
        )

        self.feat_extractor_5x5_n = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_5x5, kernel_size=5, padding=2)
        )

        self.unet = NoisePredictingUNet(self.num_features_tot, 3)


    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        features_1x1 = self.feat_extractor_1x1_n(frame_input)
        features_3x3 = self.feat_extractor_3x3_n(frame_input)
        features_5x5 = self.feat_extractor_5x5_n(frame_input)

        embedding = torch.cat((features_1x1, features_3x3, features_5x5), dim=1)

        frame_output = albedo * self.unet(embedding)

        return (frame_output, temporal_state)


class ExtendedSVGF(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using extended SVGF")

        self.num_input_channels = 12
        self.num_features_1x1 = 4
        self.num_features_3x3 = 4
        self.num_features_5x5 = 4
        self.num_features_tot = self.num_features_1x1 + self.num_features_3x3 + self.num_features_5x5
        self.num_dynamic_channels = 6 # unlike previous architectures, this includes color

        self.feat_extractor_1x1 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_1x1, kernel_size=1, padding=0)
        )

        self.feat_extractor_3x3 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_3x3, kernel_size=3, padding=1)
        )

        self.feat_extractor_5x5 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_5x5, kernel_size=5, padding=2)
        )


        self.hidden_state_builder = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, self.num_dynamic_channels - 4, kernel_size=5, padding=2),
        )

        self.alpha_calc = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels - 3 + 1, self.num_dynamic_channels - 3 + 1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels - 3 + 1, self.num_dynamic_channels - 3 + 1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels - 3 + 1, self.num_dynamic_channels - 4 + 2, kernel_size=5, padding=2),
        )

        # output denoised color for now
        self.init_filter = WeightTransformBilateralFilter(self.num_features_tot + 3, 3, 7, 1)

        self.num_filters = 3
        self.encoders = nn.ModuleList([])
        self.filters = nn.ModuleList([])
        for i in range(self.num_filters):
            dilation = (2 ** i)
            num_layer_inputs = self.num_features_tot + self.num_dynamic_channels
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(num_layer_inputs, num_layer_inputs, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.ReLU(),
                    nn.Conv2d(num_layer_inputs, 2 * num_layer_inputs + 2 - 3, kernel_size=3, padding=dilation, dilation=dilation)
                )
            )

            self.filters.append(
                BetterPixelBilateralFilter(self.num_dynamic_channels, kernel_size=5, dilation=dilation)
            )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

class ExtendedSVGF2(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using extended SVGF2")

        self.num_input_channels = 12
        self.num_features_1x1 = 4
        self.num_features_3x3 = 4
        self.num_features_5x5 = 4
        self.num_features_tot = self.num_features_1x1 + self.num_features_3x3 + self.num_features_5x5
        self.num_dynamic_channels = 4 # unlike previous architectures, this includes color

        self.batch_norm = nn.BatchNorm2d(self.num_input_channels)

        self.feat_extractor_1x1 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_1x1, kernel_size=1, padding=0)
        )

        self.feat_extractor_3x3 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_3x3, kernel_size=3, padding=1)
        )

        self.feat_extractor_5x5 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_features_5x5, kernel_size=5, padding=2)
        )


        self.hidden_state_builder = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, self.num_dynamic_channels - 4, kernel_size=5, padding=2),
        )

        self.alpha_calc = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels - 3 + 1, self.num_dynamic_channels - 3 + 1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels - 3 + 1, self.num_dynamic_channels - 3 + 1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels - 3 + 1, self.num_dynamic_channels - 4 + 2, kernel_size=5, padding=2),
        )

        # output denoised color for now
        self.init_filter = WeightTransformBilateralFilter(self.num_features_tot + 3, 3, 7, 1)

        self.num_filters = 4
        self.encoders = nn.ModuleList([])
        self.filters = nn.ModuleList([])
        for i in range(self.num_filters):
            dilation = (2 ** i)
            num_layer_inputs = self.num_features_tot + self.num_dynamic_channels
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(num_layer_inputs, 2 * num_layer_inputs + 2 - 3 + self.num_features_tot, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.ReLU(),
                    ResNeXtBlock(2 * num_layer_inputs + 2 - 3 + self.num_features_tot, 6, 7, dilation)
                )
            )

            self.filters.append(
                BetterPixelBilateralFilter2(num_layer_inputs, self.num_dynamic_channels, kernel_size=5, dilation=dilation)
            )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        norm_frame_input = self.batch_norm(frame_input)

        features_1x1 = self.feat_extractor_1x1(norm_frame_input)
        features_3x3 = self.feat_extractor_3x3(norm_frame_input)
        features_5x5 = self.feat_extractor_5x5(norm_frame_input)

        embedding = torch.cat((frame_input[:, :3, :, :], features_1x1, features_3x3, features_5x5), dim=1)
        init_color = self.init_filter(embedding)

        neighboorhood_size = 13
        padding = neighboorhood_size // 2
        neighboorhood_weight = torch.ones(1, 1, neighboorhood_size, neighboorhood_size, device=frame_input.device) / (neighboorhood_size * neighboorhood_size)

        lum = nn.functional.conv2d(init_color, torch.ones(1, 3, 1, 1, device=frame_input.device) / 3)
        mean = nn.functional.conv2d(lum, neighboorhood_weight, padding=padding)
        var = nn.functional.conv2d(lum * lum, neighboorhood_weight, padding=padding) - mean * mean

        init_color = torch.cat((init_color, var), dim=1)

        if i == 0:
            # need to build extra hidden state somehow
            hidden_state = self.hidden_state_builder(init_color)
            init_color = torch.cat((init_color, hidden_state), dim=1)
        else:
            reproj_info = torch.cat((var, temporal_state[:, 3:, :, :]), dim=1)
            alpha = self.alpha_calc(reproj_info)

            factors = nn.functional.softmax(alpha[:, :2, :, :], dim=1)
            init_color = torch.cat((init_color, alpha[:, 2:, :, :]), dim=1)

            init_color = factors[:, :1, :, :] * init_color + factors[:, 1:, :, :] * temporal_state

        # let's now do the actual filtering we came here for

        embedding = torch.cat((init_color, embedding[:, 3:, :, :]), dim=1)

        for i in range(self.num_filters):
            extract = self.encoders[i](embedding)

            packed = torch.cat((embedding[:, :3, :, :], extract[:, :-self.num_features_tot, :, :]), dim=1)

            filtered = self.filters[i](packed)

            if i != self.num_filters - 1:
                embedding = torch.cat((filtered, extract[:, -self.num_features_tot:, :, :]), dim=1)
            else:
                embedding = filtered

        temporal_state = embedding[:, :self.num_dynamic_channels, :, :]
        frame_output = albedo * embedding[:, :3, :, :]


        return (frame_output, temporal_state)




class ExtendedSVGF3(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using extended SVGF3")

        self.num_input_channels = 12
        self.num_feature_channels = 14
        self.num_dynamic_channels = 6 # unlike previous architectures, this includes color
        self.num_filters = 4

        # takes var + mean
        # predict features and hidden state
        self.preencoder = nn.Sequential(
            nn.BatchNorm2d(self.num_input_channels + 4),
            nn.Conv2d(self.num_input_channels + 4, (self.num_input_channels + 4) * 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d((self.num_input_channels + 4) * 2),
            nn.Conv2d((self.num_input_channels + 4) * 2, (self.num_input_channels + 4) * 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d((self.num_input_channels + 4) * 2),
            nn.Conv2d((self.num_input_channels + 4) * 2, self.num_feature_channels + self.num_dynamic_channels, kernel_size=1),
        )

        self.encoders = nn.ModuleList([])
        self.filters = nn.ModuleList([])
        self.num_groups = 8
        self.num_intermediate = 6
        for i in range(self.num_filters):
            # take in prev next state features, predict this state features and next state features
            self.encoders.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.num_dynamic_channels + self.num_feature_channels),
                    nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.num_dynamic_channels + self.num_feature_channels * 2),
                    nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_feature_channels * 2, kernel_size=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.num_feature_channels * 2),
                    EESPv2Block(self.num_feature_channels * 2, self.num_intermediate, self.num_groups, False),
                    nn.ReLU(),
                )
            )

            self.filters.append(
                WeightTransformKernelBilateralFilter(self.num_feature_channels + self.num_dynamic_channels, self.num_dynamic_channels, 5, False)
            )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels, self.num_dynamic_channels * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 2, self.num_dynamic_channels * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 2, 3, kernel_size=1),
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]
        color = frame_input[:, :3, :, :]

        lum = nn.functional.conv2d(color, torch.ones(1, 3, 1, 1, device=frame_input.device) / 3)

        (mean_a, var_a) = self.calc_mean_var(lum, 15)
        (mean_b, var_b) = self.calc_mean_var(lum, 25)

        embedding = self.preencoder(torch.cat((frame_input, mean_a, var_a, mean_b, var_b), dim=1))

        for i in range(self.num_filters):
            enc = self.encoders[i](embedding)

            features = enc[:, :self.num_feature_channels, :, :]
            next = enc[:, self.num_feature_channels:, :, :]

            embedding = torch.cat((embedding[:, :self.num_dynamic_channels, :, :], features), dim=1)
            filtered = self.filters[i](embedding)

            embedding = torch.cat((filtered, next), dim=1)


        frame_output = albedo * self.decoder(filtered)

        return (frame_output, temporal_state)

    def calc_mean_var(self, lum, size):
        neighboorhood_size = size
        padding = neighboorhood_size // 2
        neighboorhood_weight = torch.ones(1, 1, neighboorhood_size, neighboorhood_size, device=lum.device) / (neighboorhood_size * neighboorhood_size)

        mean = nn.functional.conv2d(lum, neighboorhood_weight, padding=padding)
        var = nn.functional.conv2d(lum * lum, neighboorhood_weight, padding=padding) - mean * mean

        return (mean, var)

class LookBackFilter(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using look back filter")

        self.num_input_channels = 12
        self.num_channels = 12
        self.num_dynamic_channels = 4

        self.preprocess = nn.BatchNorm2d(self.num_input_channels - 3)

        self.lookahead_unet = LookAheadUNet(self.num_input_channels, self.num_channels)

        self.filter0 = WeightTransformBilateralFilter(self.num_channels, self.num_dynamic_channels, 5, 1)

        self.enc1_0 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels * 2, kernel_size=5, padding=2, dilation=1)
        )
        # add skip here
        self.enc1_1 = nn.Sequential(
            nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=5, padding=4, dilation=2),
        )

        self.filter1 = WeightTransformBilateralFilter(self.num_channels * 2, self.num_dynamic_channels * 2, 5, 2)

        self.enc2_0 = nn.Sequential(
            nn.Conv2d(self.num_channels * 2, self.num_channels * 4, kernel_size=5, padding=4, dilation=2)
        )
        # add skip here
        self.enc2_1 = nn.Sequential(
            nn.Conv2d(self.num_channels * 4, self.num_channels * 4, kernel_size=5, padding=8, dilation=4),
            nn.ReLU(),
            nn.Conv2d(self.num_channels * 4, self.num_channels * 4, kernel_size=5, padding=8, dilation=4),
        )

        self.filter2 = WeightTransformBilateralFilter(self.num_channels * 4, self.num_dynamic_channels * 4, 5, 4)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels * 4, self.num_dynamic_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 4, self.num_dynamic_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 4, 3, kernel_size=3, padding=1),
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.preprocess(frame_input[:, 3:, :, :])
        frame_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        (x6, x5, x4) = self.lookahead_unet(frame_input)

        embedding0 = x6
        filtered0 = self.filter0(embedding0)
        filtered0 = torch.cat((filtered0, embedding0[:, self.num_dynamic_channels:, :, :]), dim=1)

        embedding1 = self.enc1_0(filtered0) + x5
        embedding1 = self.enc1_1(embedding1)
        filtered1 = self.filter1(embedding1)
        filtered1 = torch.cat((filtered1, embedding1[:, self.num_dynamic_channels * 2:, :, :]), dim=1)

        embedding2 = self.enc2_0(filtered1) + x4
        embedding2 = self.enc2_1(embedding2)
        filtered2 = self.filter2(embedding2)

        frame_output = albedo * self.decoder(filtered2)

        return (frame_output, temporal_state)



class GlobalContextPreEncodingFilter(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using GlobalContextPreEncodingFilter")

        self.num_input_channels = 12
        self.num_feature_channels = 12
        self.num_dynamic_channels = 6
        self.num_filters = 3

        self.preprocess = nn.BatchNorm2d(self.num_input_channels - 3)

        self.preencoder = PreEncodingUNet3(self.num_input_channels, self.num_feature_channels, 4, self.num_filters)

        self.enc0 = nn.Sequential(
            nn.Conv2d(self.num_input_channels + self.num_feature_channels, self.num_input_channels + self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + self.num_dynamic_channels + self.num_feature_channels * 2, self.num_input_channels + self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + self.num_dynamic_channels + self.num_feature_channels * 2, self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=3, padding=1, dilation=1),
        )

        self.filter0 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 1)

        self.enc1 = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_feature_channels * 2, kernel_size=3, padding=2, dilation=2),
        )
        self.filter1 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_feature_channels, kernel_size=3, padding=4, dilation=4),
        )
        self.filter2 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 4)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels, self.num_dynamic_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 2, self.num_dynamic_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 2, 3, kernel_size=3, padding=1),
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.preprocess(frame_input[:, 3:, :, :])
        frame_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        context = self.preencoder(frame_input)

        context0 = context[:, :self.num_feature_channels, :, :]
        embedding0 = self.enc0(torch.cat((frame_input, context0), dim=1))
        next0 = embedding0[:, self.num_dynamic_channels + self.num_feature_channels:, :, :]
        filtered0 = self.filter0(embedding0[:, :self.num_dynamic_channels + self.num_feature_channels])

        context1 = context[:, self.num_feature_channels:self.num_feature_channels * 2, :, :]
        embedding1 = self.enc1(torch.cat((filtered0, next0, context1), dim=1))
        next1 = embedding1[:, self.num_feature_channels:, :, :]
        filtered1 = self.filter1(torch.cat((filtered0, embedding1[:, :self.num_feature_channels]), dim=1))

        context2 = context[:, self.num_feature_channels * 2:, :, :]
        embedding2 = self.enc2(torch.cat((filtered1, next1, context2), dim=1))
        filtered2 = self.filter2(torch.cat((filtered1, embedding2), dim=1))


        frame_output = albedo * self.decoder(filtered2)

        return (frame_output, temporal_state)






class GlobalContextPreEncodingFilter2(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using GlobalContextPreEncodingFilter2")

        self.num_input_channels = 12
        self.num_feature_channels = 12
        self.num_dynamic_channels = 4
        self.num_filters = 3

        self.preprocess = nn.BatchNorm2d(self.num_input_channels - 3)

        self.preencoder = PreEncodingUNet2(self.num_input_channels, self.num_feature_channels, self.num_dynamic_channels, self.num_filters)

        self.filter0 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 1)
        self.filter1 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 2)
        self.filter2 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 4)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels, self.num_dynamic_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 2, self.num_dynamic_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 2, 3, kernel_size=3, padding=1),
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.preprocess(frame_input[:, 3:, :, :])
        frame_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        context = self.preencoder(frame_input)

        embedding0 = context[:, :self.num_dynamic_channels + self.num_feature_channels, :, :]
        filtered0 = self.filter0(embedding0)

        context1 = context[:, self.num_dynamic_channels + self.num_feature_channels:self.num_dynamic_channels + self.num_feature_channels * 2, :, :]
        embedding1 = torch.cat((filtered0, context1), dim=1)
        filtered1 = self.filter1(embedding1)

        context2 = context[:, self.num_dynamic_channels + self.num_feature_channels * 2:, :, :]
        embedding2 = torch.cat((filtered1, context2), dim=1)
        filtered2 = self.filter2(embedding2)

        frame_output = albedo * self.decoder(filtered2)

        return (frame_output, temporal_state)



class GlobalContextPreEncodingFilter3(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using GlobalContextPreEncodingFilter3")

        self.num_input_channels = 12
        self.num_feature_channels = 12
        self.num_dynamic_channels = 6
        self.num_filters = 3

        self.preprocess = nn.BatchNorm2d(self.num_input_channels - 3)

        self.preencoder = PreEncodingUNet3(self.num_input_channels, self.num_feature_channels, 4, self.num_filters)

        self.enc0 = nn.Sequential(
            nn.Conv2d(self.num_input_channels + self.num_feature_channels, self.num_dynamic_channels + self.num_feature_channels * 2, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            ResNeXtBlock(self.num_dynamic_channels + self.num_feature_channels * 2, 6, 5, 1)
        )

        self.filter0 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 1)

        self.enc1 = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_feature_channels * 2, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            ResNeXtBlock(self.num_feature_channels * 2, 6, 5, 2)
        )
        self.filter1 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels + self.num_feature_channels * 2, self.num_feature_channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            ResNeXtBlock(self.num_feature_channels, 6, 5, 4)
        )
        self.filter2 = WeightTransformBilateralFilter(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 4)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_dynamic_channels, self.num_dynamic_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 2, self.num_dynamic_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_dynamic_channels * 2, 3, kernel_size=3, padding=1),
        )

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.preprocess(frame_input[:, 3:, :, :])
        frame_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        context = self.preencoder(frame_input)

        context0 = context[:, :self.num_feature_channels, :, :]
        embedding0 = self.enc0(torch.cat((frame_input, context0), dim=1))
        next0 = embedding0[:, self.num_dynamic_channels + self.num_feature_channels:, :, :]
        filtered0 = self.filter0(embedding0[:, :self.num_dynamic_channels + self.num_feature_channels])

        context1 = context[:, self.num_feature_channels:self.num_feature_channels * 2, :, :]
        embedding1 = self.enc1(torch.cat((filtered0, next0, context1), dim=1))
        next1 = embedding1[:, self.num_feature_channels:, :, :]
        filtered1 = self.filter1(torch.cat((filtered0, embedding1[:, :self.num_feature_channels]), dim=1))

        context2 = context[:, self.num_feature_channels * 2:, :, :]
        embedding2 = self.enc2(torch.cat((filtered1, next1, context2), dim=1))
        filtered2 = self.filter2(torch.cat((filtered1, embedding2), dim=1))


        frame_output = albedo * self.decoder(filtered2)

        return (frame_output, temporal_state)



class GlobalContextPreEncodingFilter4(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using GlobalContextPreEncodingFilter4 (EESP Edition)")

        self.num_input_channels = 12
        self.num_feature_channels = 12
        self.num_channels_per_filter = self.num_feature_channels + 1
        self.num_dynamic_channels = 3 # TIED TO COLOR!
        self.num_filters = 4

        self.preprocess = nn.BatchNorm2d(self.num_input_channels - 3)

        self.preencoder = EESPPreencoder(self.num_input_channels, self.num_channels_per_filter, 6, self.num_filters, 5, 9)

        self.filter = nn.ModuleList([])
        for i in range(self.num_filters):
            self.filter.append(
                LocalAttention(self.num_dynamic_channels + self.num_feature_channels, self.num_dynamic_channels, 5, 2 ** i)
            )


        self.unfold = nn.Unfold(kernel_size=(3, 3), padding=(1, 1))
        self.upsample_kernel_gen = nn.Sequential(
            nn.Conv2d(self.num_input_channels - 3, 9, kernel_size=1),
            nn.ReLU(),
            ResNeXtBlock(9, 12, 3, 1),
        )

        self.downsample = True

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

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
        albedo = frame_input[:, 3:6, :, :]

        if self.downsample:
            downsampled = nn.functional.avg_pool2d(frame_input, kernel_size=2)
        else:
            downsampled = frame_input

        lum = nn.functional.conv2d(downsampled[:, :3, :, :], torch.ones(1, 3, 1, 1, device=downsampled.device) / 3)
        (mean_a, var_a) = self.calc_mean_var(lum, 7)
        (mean_b, var_b) = self.calc_mean_var(lum, 15)

        norm_aux = self.preprocess(downsampled[:, 3:, :, :])
        preenc_input = torch.cat((downsampled[:, :3, :, :], norm_aux, mean_a, var_a, mean_b, var_b), dim=1)

        global_context = self.preencoder(preenc_input)

        blended = downsampled[:, :3, :, :]
        for i in range(self.num_filters):
            context_start = i * self.num_channels_per_filter
            context_end = context_start + self.num_channels_per_filter

            context = global_context[:, context_start:context_end, :, :]
            embedding = torch.cat((blended, context[:, :-1, :, :]), dim=1)

            filtered = self.filter[i](embedding)
            weight = nn.functional.sigmoid(context[:, -1:, :, :])

            blended = filtered * weight + blended * (1 - weight)

        if self.downsample:
            unfold = self.unfold(blended).view(downsampled.size(0), 27, downsampled.size(2), downsampled.size(3))

            upsampled = nn.functional.upsample(unfold, scale_factor=2, mode="bilinear", align_corners=True)

            upsampled = upsampled.view(frame_input.size(0), 3, 9, frame_input.size(2), frame_input.size(3))

            scores = self.upsample_kernel_gen(frame_input[:, 3:, :, :])
            kernels = scores.unsqueeze(1)

            final = (upsampled * kernels).sum(2)
        else:
            final = blended


        frame_output = albedo * final

        return (frame_output, temporal_state)

    def calc_mean_var(self, lum, size):
        neighboorhood_size = size
        padding = neighboorhood_size // 2
        neighboorhood_weight = torch.ones(1, 1, neighboorhood_size, neighboorhood_size, device=lum.device) / (neighboorhood_size * neighboorhood_size)

        mean = nn.functional.conv2d(lum, neighboorhood_weight, padding=padding)
        var = nn.functional.conv2d(lum * lum, neighboorhood_weight, padding=padding) - mean * mean

        return (mean, var)



class ImageEnhancementFilter(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using ImageEnhancementFilter (EESP Edition)")

        self.num_input_channels = 12
        self.num_feature_channels = 8

        self.preprocess = nn.BatchNorm2d(self.num_input_channels - 3)

        # we use this not because it predicts noise but it is very heavy weight
        self.transform_predictor =  nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels, self.num_input_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),
            EESPPreencoder2(self.num_input_channels, 3 * (self.num_feature_channels + 1), 16, 1, 4),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )
        # TransformPredictingUNet(self.num_input_channels, 3 * (self.num_feature_channels + 1))#

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels - 3, self.num_input_channels - 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels - 3, self.num_input_channels - 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels - 3, self.num_input_channels - 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels - 3, self.num_feature_channels, kernel_size=3, padding=1),
        )

        self.batch_norm = nn.BatchNorm2d(self.num_input_channels - 3)

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.preprocess(frame_input[:, 3:, :, :])
        norm_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        # downsample our input
        transforms = self.transform_predictor(norm_input).permute((0, 2, 3, 1))
        transforms = transforms.view(B, H, W, 3, -1)

        features = self.feature_encoder(norm_aux).permute((0, 2, 3, 1)).unsqueeze(4)
        res = torch.matmul(transforms[:, :, :, :, :-1], features) + transforms[:, :, :, :, -1:]

        res = res.flatten(3).permute((0, 3, 1, 2))

        frame_output = albedo * res

        return (frame_output, temporal_state)



class ImageEnhancementFilter(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using ImageEnhancementFilter (EESP Edition)")

        self.num_input_channels = 12
        self.num_feature_channels = 8

        self.transform_predictor =  nn.Sequential(
            nn.Conv2d(self.num_input_channels + 3, self.num_input_channels + 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels + 3, self.num_input_channels + 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),
            EESPPreencoder2(self.num_input_channels + 3, 3 * (self.num_feature_channels + 1) + 1, 16, 1, 4),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels - 3, self.num_input_channels - 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels - 3, self.num_input_channels - 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels - 3, self.num_input_channels - 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_input_channels - 3, self.num_feature_channels, kernel_size=3, padding=1),
        )

        self.batch_norm = nn.BatchNorm2d(self.num_input_channels - 3)

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        if temporal_state is not None:
            prev_color, prev_transform = temporal_state
        else:
            prev_color = torch.zeros_like(frame_input[:, :3, :, :])
            prev_transform = torch.zeros(B, 3 * (self.num_feature_channels + 1), H, W, device=frame_input.device)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.batch_norm(frame_input[:, 3:, :, :])
        norm_input = torch.cat((frame_input[:, :3, :, :], prev_color, norm_aux), dim=1)

        # downsample our input
        packed = self.transform_predictor(norm_input)
        alpha = torch.sigmoid(packed[:, -1:, :, :])
        transforms = packed[:, :-1, :, :] * alpha + prev_transform * (1 - alpha)

        prev_transform = transforms

        transforms = transforms.permute((0, 2, 3, 1)).view(B, H, W, 3, -1)

        features = self.feature_encoder(norm_aux).permute((0, 2, 3, 1)).unsqueeze(4)
        res = torch.matmul(transforms[:, :, :, :, :-1], features) + transforms[:, :, :, :, -1:]

        res = res.flatten(3).permute((0, 3, 1, 2))

        temporal_state = (res, prev_transform)

        frame_output = albedo * res

        return (frame_output, temporal_state)

"""
KQ^T
(WK+B)(WQ+B)^T
(WK+B)(WQ)^T + (WK+B)B^T
(WK+B)(Q^T x W^T) + WKB^T + ||B||^2
WKQ^T x W^T + BQ^T x W^T
"""


class TransformerPredictingHRF(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using TransformerPredictingHRF (EESP Edition)")

        self.num_input_channels = 12
        self.num_feature_channels = 6

        self.num_kernel_channels_per_layer = 3 * 3 + 1
        self.num_w_channels_per_layer = self.num_feature_channels * (self.num_feature_channels + 1)
        self.num_filter_channels_per_layer = self.num_kernel_channels_per_layer + self.num_w_channels_per_layer

        self.num_filter_layers = 4
        self.num_tot_filter_channels = self.num_filter_layers * self.num_filter_channels_per_layer

        self.transform_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            TransformPredictingUNet2(self.num_input_channels, self.num_tot_filter_channels + self.num_feature_channels),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )
        self.batch_norm = nn.BatchNorm2d(self.num_input_channels - 3)

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.batch_norm(frame_input[:, 3:, :, :])
        norm_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        # transforms is [features, layer 0 info, layer 1 info, etc]
        transforms = checkpoint.checkpoint_sequential(self.transform_predictor, segments=1, input=norm_input, use_reentrant=False)

        features = transforms[:, :self.num_feature_channels, :, :]

        color = frame_input[:, :3, :, :]
        for i in range(self.num_filter_layers):
            start = self.num_feature_channels + self.num_filter_channels_per_layer * i
            layer_weights = transforms[:, start:start + self.num_filter_channels_per_layer, :, :]
            color = checkpoint.checkpoint(self.filter_layer, color, features, layer_weights, i, use_reentrant=False)

        frame_output = albedo * color

        return (frame_output, temporal_state)

    def filter_layer(self, color, features, network, layer_idx):
        B = color.size(0)
        H = color.size(2)
        W = color.size(3)

        # extract parameters
        kernel = network[:, :10, :, :]
        wk = network[:, 10:10 + self.num_w_channels_per_layer, :, :].view(B, self.num_feature_channels + 1, self.num_feature_channels, H, W).permute((0, 3, 4, 1, 2))
        wq = wk#network[:, 10 + self.num_w_channels_per_layer:, :, :].view(B, self.num_feature_channels + 1, self.num_feature_channels, H, W).permute((0, 3, 4, 1, 2))

        dilation = 2 ** layer_idx

        compressed_features = nn.functional.interpolate(nn.functional.avg_pool2d(features, kernel_size=dilation, stride=dilation), scale_factor=dilation, mode="bilinear", align_corners=True)
        compressed_color = nn.functional.interpolate(nn.functional.avg_pool2d(color, kernel_size=dilation, stride=dilation), scale_factor=dilation, mode="bilinear", align_corners=True)

        feature_window = nn.functional.unfold(compressed_features, kernel_size=3, padding=dilation, dilation=dilation).view(B, self.num_feature_channels, 9, H, W)
        color_window = nn.functional.unfold(compressed_color, kernel_size=3, padding=dilation, dilation=dilation).view(B, 3, 9, H, W)

        # append center pixel to weight

        # (B, F, 9, H, W) -> (B, F, 10, H, W)
        feature_window = torch.cat((feature_window, features.unsqueeze(2)), dim=2)
        color_window = torch.cat((color_window, color.unsqueeze(2)), dim=2)

        # (B, F, 10, H, W) -> (B, H, W, 10, F)
        feature_window = feature_window.permute((0, 3, 4, 2, 1))
        color_window = color_window.permute((0, 3, 4, 2, 1))

        # (B, H, W, F) -> (B, H, W, 1, F)
        center_feature = features.permute((0, 2, 3, 1)).unsqueeze(3)

        # (B, H, W, 10, F) -> same
        k = torch.matmul(feature_window, wk[:, :, :, :-1, :]) + wk[:, :, :, -1:, :]
        # (B, H, W, 1, F) -> same
        q = torch.matmul(center_feature, wq[:, :, :, :-1, :]) + wq[:, :, :, -1:, :]


        # (B, H, W, 1, F) -> (B, H, W, F, 1)
        # (B, H, W, 10, F) x (B, H, W, F, 1) = (B, H, W, 10, 1)
        scores = torch.matmul(k, q.transpose(3, 4)) / math.sqrt(10.0)

        # (B, 10, H, W) -> (B, H, W, 10, 1)
        # (B, H, W, 10, 1) -> same
        scores = scores * kernel.permute((0, 2, 3, 1)).unsqueeze(4)

        # (B, H, W, 10, 1) -> same
        softmax = nn.functional.softmax(scores, dim=3)

        # (B, H, W, 10, 1) * (B, H, W, 10, 3) = (B, H, W, 10, 3)
        # (B, H, W, 10, 3) -> (B, H, W, 3)
        clustered = (softmax * color_window).sum(3)

        # (B, H, W, 3) -> (B, 3, H, W)
        reordered = clustered.permute((0, 3, 1, 2))

        return reordered



class TransformerPredictingHRF2(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using TransformerPredictingHRF2")

        self.num_input_channels = 12
        self.num_feature_channels = 6

        self.num_kernel_channels_per_layer = 3 * 3 + 1
        self.num_feature_channels_per_layer = self.num_feature_channels * 2
        self.num_filter_channels_per_layer = self.num_kernel_channels_per_layer + self.num_feature_channels_per_layer

        self.num_filter_layers = 4
        self.num_tot_filter_channels = self.num_filter_layers * self.num_filter_channels_per_layer

        self.transform_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            TransformPredictingUNet2(self.num_input_channels, self.num_tot_filter_channels + self.num_feature_channels),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )
        self.batch_norm = nn.BatchNorm2d(self.num_input_channels - 3)

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.batch_norm(frame_input[:, 3:, :, :])
        norm_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        # transforms is [features, layer 0 info, layer 1 info, etc]
        transforms = checkpoint.checkpoint_sequential(self.transform_predictor, segments=1, input=norm_input, use_reentrant=False)

        color = frame_input[:, :3, :, :]
        for i in range(self.num_filter_layers):
            start = self.num_filter_channels_per_layer * i
            network = transforms[:, start:start + self.num_filter_channels_per_layer, :, :]
            color = checkpoint.checkpoint(self.filter_layer, color, network, i, use_reentrant=False)

        frame_output = albedo * color

        return (frame_output, temporal_state)

    def filter_layer(self, color, network, layer_idx):
        B = color.size(0)
        H = color.size(2)
        W = color.size(3)

        # extract parameters
        kernel = network[:, :10, :, :]
        key_features = network[:, 10:10 + self.num_feature_channels, :, :]
        query_features = network[:, 10 + self.num_feature_channels:, :, :]

        dilation = 2 ** layer_idx

        compressed_features = key_features#nn.functional.interpolate(nn.functional.avg_pool2d(key_features, kernel_size=dilation, stride=dilation), scale_factor=dilation, mode="bilinear", align_corners=True)
        compressed_color = color#nn.functional.interpolate(nn.functional.avg_pool2d(color, kernel_size=dilation, stride=dilation), scale_factor=dilation, mode="bilinear", align_corners=True)

        feature_window = nn.functional.unfold(compressed_features, kernel_size=3, padding=dilation, dilation=dilation).view(B, self.num_feature_channels, 9, H, W)
        color_window = nn.functional.unfold(compressed_color, kernel_size=3, padding=dilation, dilation=dilation).view(B, 3, 9, H, W)

        # append center pixel to weight

        # (B, F, 9, H, W) -> (B, F, 10, H, W)
        feature_window = torch.cat((feature_window, key_features.unsqueeze(2)), dim=2)
        color_window = torch.cat((color_window, color.unsqueeze(2)), dim=2)

        # (B, F, 10, H, W) -> (B, H, W, 10, F)
        feature_window = feature_window.permute((0, 3, 4, 2, 1))
        color_window = color_window.permute((0, 3, 4, 2, 1))

        # (B, H, W, F) -> (B, H, W, 1, F)
        center_feature = query_features.permute((0, 2, 3, 1)).unsqueeze(3)

        # (B, H, W, 10, F) -> same
        k = feature_window
        # (B, H, W, 1, F) -> same
        q = center_feature


        # (B, H, W, 1, F) -> (B, H, W, F, 1)
        # (B, H, W, 10, F) x (B, H, W, F, 1) = (B, H, W, 10, 1)
        scores = torch.matmul(k, q.transpose(3, 4)) / math.sqrt(10.0)

        # (B, 10, H, W) -> (B, H, W, 10, 1)
        # (B, H, W, 10, 1) -> same
        scores = scores * kernel.permute((0, 2, 3, 1)).unsqueeze(4)

        # (B, H, W, 10, 1) -> same
        softmax = nn.functional.softmax(scores, dim=3)

        # (B, H, W, 10, 1) * (B, H, W, 10, 3) = (B, H, W, 10, 3)
        # (B, H, W, 10, 3) -> (B, H, W, 3)
        clustered = (softmax * color_window).sum(3)

        # (B, H, W, 3) -> (B, 3, H, W)
        reordered = clustered.permute((0, 3, 1, 2))

        return reordered




class TransformerPredictingHRF3(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using TransformerPredictingHRF3")

        self.num_input_channels = 12
        self.num_feature_channels = 6

        self.num_kernel_channels_per_layer = 3 * 3 + 1
        self.num_feature_channels_per_layer = self.num_feature_channels * 2
        self.num_filter_channels_per_layer = self.num_kernel_channels_per_layer + self.num_feature_channels_per_layer

        self.num_filter_layers = 4
        self.num_tot_filter_channels = self.num_filter_layers * self.num_filter_channels_per_layer

        self.transform_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            TransformPredictingUNet2(self.num_input_channels, self.num_tot_filter_channels + self.num_feature_channels),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )
        self.batch_norm = nn.BatchNorm2d(self.num_input_channels - 3)

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.batch_norm(frame_input[:, 3:, :, :])
        norm_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        # transforms is [features, layer 0 info, layer 1 info, etc]
        transforms = checkpoint.checkpoint_sequential(self.transform_predictor, segments=1, input=norm_input, use_reentrant=False)

        color = frame_input[:, :3, :, :]
        for i in range(self.num_filter_layers):
            start = self.num_filter_channels_per_layer * i
            network = transforms[:, start:start + self.num_filter_channels_per_layer, :, :]
            color = checkpoint.checkpoint(self.filter_layer, color, network, i, use_reentrant=False)

        frame_output = albedo * color

        return (frame_output, temporal_state)

    def filter_layer(self, color, network, layer_idx):
        B = color.size(0)
        H = color.size(2)
        W = color.size(3)

        # extract parameters
        kernel = network[:, :10, :, :]
        key_features = network[:, 10:10 + self.num_feature_channels, :, :]
        query_features = network[:, 10 + self.num_feature_channels:, :, :]

        dilation = 2 ** layer_idx

        compressed_features = key_features#nn.functional.interpolate(nn.functional.avg_pool2d(key_features, kernel_size=dilation, stride=dilation), scale_factor=dilation, mode="bilinear", align_corners=True)
        compressed_color = color#nn.functional.interpolate(nn.functional.avg_pool2d(color, kernel_size=dilation, stride=dilation), scale_factor=dilation, mode="bilinear", align_corners=True)

        feature_window = nn.functional.unfold(compressed_features, kernel_size=3, padding=dilation, dilation=dilation).view(B, self.num_feature_channels, 9, H, W)
        color_window = nn.functional.unfold(compressed_color, kernel_size=3, padding=dilation, dilation=dilation).view(B, 3, 9, H, W)

        # append center pixel to weight

        # (B, F, 9, H, W) -> (B, F, 10, H, W)
        feature_window = torch.cat((feature_window, key_features.unsqueeze(2)), dim=2)
        color_window = torch.cat((color_window, color.unsqueeze(2)), dim=2)

        # (B, F, 10, H, W) -> (B, H, W, 10, F)
        feature_window = feature_window.permute((0, 3, 4, 2, 1))
        color_window = color_window.permute((0, 3, 4, 2, 1))

        # (B, H, W, F) -> (B, H, W, 1, F)
        center_feature = query_features.permute((0, 2, 3, 1)).unsqueeze(3)

        # (B, H, W, 10, F) -> same
        k = feature_window
        # (B, H, W, 1, F) -> same
        q = center_feature


        # (B, H, W, 1, F) -> (B, H, W, F, 1)
        # (B, H, W, 10, F) x (B, H, W, F, 1) = (B, H, W, 10, 1)
        scores = torch.matmul(k, q.transpose(3, 4)) / math.sqrt(10.0)

        # (B, 10, H, W) -> (B, H, W, 10, 1)
        # (B, H, W, 10, 1) -> same
        scores = scores * kernel.permute((0, 2, 3, 1)).unsqueeze(4)

        # (B, H, W, 10, 1) -> same
        softmax = nn.functional.softmax(scores, dim=3)

        # (B, H, W, 10, 1) * (B, H, W, 10, 3) = (B, H, W, 10, 3)
        # (B, H, W, 10, 3) -> (B, H, W, 3)
        clustered = (softmax * color_window).sum(3)

        # (B, H, W, 3) -> (B, 3, H, W)
        reordered = clustered.permute((0, 3, 1, 2))

        return reordered




class LocalFrequencyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using LocalFrequencyDenoiser")

        self.num_input_channels = 12
        self.num_frequency_channels = 5
        self.num_frequencies = 4

        self.frequency_predictor = TransformPredictingUNet(self.num_input_channels, self.num_frequency_channels * self.num_frequencies)
        self.batch_norm = nn.BatchNorm2d(self.num_input_channels - 3)

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.batch_norm(frame_input[:, 3:, :, :])
        norm_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        raw_frequencies = self.frequency_predictor(norm_input).view(B, self.num_frequencies, self.num_frequency_channels, H, W)


        xpos = torch.arange(0, H, 1, device=frame_input.device).view(1, 1, H, 1).expand(B, -1, -1, W)
        ypos = torch.arange(0, W, 1, device=frame_input.device).view(1, 1, 1, W).expand(B, -1, H, -1)

        posenc = torch.cat((xpos, ypos, 2 * xpos + ypos, xpos + 2 * ypos), dim=1)

        posenc = torch.sin(posenc / 30.0)

        wave = torch.sin(raw_frequencies[:, :, 0, :, :] * posenc + raw_frequencies[:, :, 1, :, :]).unsqueeze(2)

        reconstructed = (wave * raw_frequencies[:, :, 2:, :, :]).sum(1)

        recolored = albedo * reconstructed

        return (recolored, temporal_state)



class LocalFrequencyDenoiser2(nn.Module):
    def __init__(self):
        super().__init__()

        print("Using LocalFrequencyDenoiser2")

        self.num_input_channels = 12
        self.num_frequency_channels = 4
        self.num_frequencies = 4

        self.frequency_predictor = TransformPredictingUNet(self.num_input_channels, self.num_frequency_channels * self.num_frequencies)
        self.batch_norm = nn.BatchNorm2d(self.num_input_channels - 3)

    def forward(self, input):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, i, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output

    def run_frame(self, frame_input, temporal_state, i):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        norm_aux = self.batch_norm(frame_input[:, 3:, :, :])
        norm_input = torch.cat((frame_input[:, :3, :, :], norm_aux), dim=1)

        raw_frequencies = self.frequency_predictor(norm_input).view(B, self.num_frequencies, self.num_frequency_channels, H, W)

        wave = torch.sin(raw_frequencies[:, :, 0, :, :]).unsqueeze(2)
        reconstructed = (wave * raw_frequencies[:, :, 1:, :, :]).sum(1)

        recolored = albedo * reconstructed

        return (recolored, temporal_state)
