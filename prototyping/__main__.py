import torch
import numpy as np
import simple_kernel
import kpcnn

# We need this so OpenCV imports exr files
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Utilizing {torch.cuda.get_device_name(0)} for training and inference.")
else:
    print("Utiilzing CPU for training and inference.")

reference = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Reference.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
color = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Color.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
albedo = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Albedo.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
worldpos = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-WorldPosition.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
worldnorm = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-WorldNormal.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

def print_shape(name, img):
    print(f"{name}\tshape is {img.shape}")

print_shape("Ref", reference)
print_shape("Color", reference)
print_shape("Albedo", reference)
print_shape("Pos", reference)
print_shape("Norm", reference)

#print_shape("Raw in", raw_in)


model = kpcnn.MiniKPCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = torch.nn.L1Loss()



target = torch.tensor(reference, device=device).permute((2, 0, 1))
target = target[None, :]

numIters = 1000
for i in range(0, numIters):
    optimizer.zero_grad()

    raw_in = np.concatenate((color, albedo, worldpos, worldnorm), axis=2)

    input = torch.tensor(raw_in, device=device).permute((2, 0, 1))
    input = input[None, :]

    output = model(input)

    loss = loss_fn(output, target)
    loss.backward()

    optimizer.step()

    print(f"Loss at iteration {i}\tis {loss.item()}")

    if i == 0 or i == (numIters - 1):
        # first, export our model
        if i == (numIters - 1):
            traced = torch.jit.trace(model, input)
            traced.to("cpu")
            traced.save("C:/FalcorFiles/Models/MiniKPCNN-2.pt")

        image = output.detach().squeeze().permute((1, 2, 0)).cpu().numpy()
        print(f"Output shape is now {image.shape}")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



