import torch
import numpy as np
import simple_kernel
import kpcnn
import frame_data
from torch.utils.data import DataLoader
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Utilizing {torch.cuda.get_device_name(0)} for training and inference.")
else:
    print("Utiilzing CPU for training and inference.")

training_data = frame_data.FrameData("C:\\FalcorFiles\\Dataset0\\", device)
training_loader = DataLoader(training_data, batch_size=1, shuffle=True)

model = kpcnn.MiniKPCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = torch.nn.L1Loss()

numIters = 1000
for i in range(0, numIters):
    input, target = next(iter(training_loader))

    optimizer.zero_grad()

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



