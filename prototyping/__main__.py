import torch
import simple_kernel

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
print(f"Ref shape is {reference.shape}")

model = simple_kernel.SimpleKernel().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
loss_fn = torch.nn.L1Loss()

input = torch.tensor(reference, device=device).permute((2, 0, 1))
input = input[None, :]

numIters = 10000
for i in range(0, numIters):
    optimizer.zero_grad()

    output = model(input)

    loss = loss_fn(output, input)
    loss.backward()

    optimizer.step()

    print(f"Loss at iteration {i}\tis {loss.item()}")

    if i == 0 or i == (numIters - 1):
        image = output.detach().squeeze().permute((1, 2, 0)).cpu().numpy()
        print(f"Output shape is now {image.shape}")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()






