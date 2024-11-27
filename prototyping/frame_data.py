import torch
import numpy as np

# We need this so OpenCV imports exr files
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2

class FrameData:
    def __init__(self, dataset_dir, device):
        self.dataset_dir=dataset_dir

        self.num_frames=0
        while True:
            if os.path.exists(self.dataset_dir + str(self.num_frames) + "-Reference.exr"):
                self.num_frames+=1
            else:
                break

        # cap everything to 1 for now
        if self.num_frames > 1:
            self.num_frames = 1

        print(f"Dataset at {self.dataset_dir} has {self.num_frames} images")

        self.device=device

    def __len__(self):
        # length here is defined by the number of frame sequneces we have,
        # not the number of frames
        return 1

    # this needs to manually convert things to a tensor
    def __getitem__(self, idx):
        # basically, I want to output it in this format for each element of each batch
        # (B, N, C, H, W)
        # B = batch size
        # N = number of frames
        # C = channels for each frame (color, albedo, world pos, world norm)

        color = self.read_exr(idx, "Color")
        albedo = self.read_exr(idx, "Albedo")
        worldpos = self.read_exr(idx, "WorldPosition")
        worldnorm = self.read_exr(idx, "WorldNormal")
        raw_in = np.concatenate((color, albedo, worldpos, worldnorm), axis=2)
        input = torch.tensor(raw_in, device=self.device).permute((2, 0, 1))

        reference = self.read_exr(idx, "Reference")
        target = torch.tensor(reference, device=self.device).permute((2, 0, 1))

        return input, target

    def read_exr(self, idx, ext):
        return cv2.imread(self.dataset_dir + str(idx) + "-" + ext + ".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    def print_shape(name, img):
        print(f"{name}\tshape is {img.shape}")
