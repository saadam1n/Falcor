import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import frame_data
from filter import *

attn = LocalAttention(1, 1, kernel_size=5, dilation=1)

input = torch.ones(1, 1, 10, 10)

print(attn(input))






fd = frame_data.FrameData("C:\\FalcorFiles\\Dataset0\\", "cpu", 1)

(frame, ref) = fd.get_full_img()

color = frame[None, :3, :, :]
albedo = frame[None, 3:6, :, :]

color = F.conv2d(color, torch.ones(3, 1, 4, 4) / 16, stride=4, groups=3)
print(f"Shape is {color.shape}")
color = F.upsample(color, scale_factor=4, mode="bilinear", align_corners=True)
print(f"Shape is {color.shape}")

image = albedo * color

image = image[0, :, :, :]
image = image.detach()
image = image.permute((1, 2, 0)).cpu().numpy()
image = image[:, :, -3:]


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
