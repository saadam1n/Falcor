import torch
import numpy as np
import random

# We need this so OpenCV imports exr files
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2

class FrameData:
    def __init__(self, dataset_dir, device, seq_len):
        self.dataset_dir=dataset_dir
        self.seq_len=seq_len

        self.num_frames=0
        while True:
            if os.path.exists(self.dataset_dir + str(self.num_frames) + "-Reference.exr"):
                self.num_frames+=1
            else:
                break

        print(f"Dataset at {self.dataset_dir} has {self.num_frames} images")

        self.device=device

        self.data_cache = {}

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

        if idx in self.data_cache:
            return self.data_cache.get(idx)

        self.yoff = 200#random.randint(0, 800)
        self.xoff = 200#random.randint(0, 1200)

        for i in range(self.seq_len):
            color = self.read_exr(idx, "Color")
            albedo = self.read_exr(idx, "Albedo")

            albedo[albedo < 0.001] = 1.0
            color = color / albedo

            worldpos = self.read_exr(idx, "WorldPosition")
            worldnorm = self.read_exr(idx, "WorldNormal")

            frame_input = torch.concat((color, albedo, worldpos, worldnorm), dim=2).permute((2, 0, 1))
            frame_reference = self.read_exr(idx, "Reference").permute((2, 0, 1))

            if i == 0:
                input = frame_input
                reference = frame_reference
            else:
                input = torch.concat((input, frame_input), dim=0)
                reference = torch.concat((reference, frame_reference), dim=0)

        #self.data_cache[idx] = input, reference

        return input, reference

    def read_exr(self, idx, ext):
        filename = str(idx) + "-" + ext + ".exr"

        cache_path = "C:/FalcorFiles/CacheV2/" + filename + ".npy"

        if os.path.exists(cache_path):
            img = np.load(cache_path)
        else:
            img = cv2.imread(self.dataset_dir + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
            np.save(cache_path, img)



        img = img[self.yoff:self.yoff+720 // 2, self.xoff:self.xoff+1280 // 2, :]

        return torch.tensor(img, device=self.device, dtype=torch.float32)

    def print_shape(name, img):
        print(f"{name}\tshape is {img.shape}")
