import torch
import numpy as np
import simple_kernel
import kpcnn
import fpcnn
import frame_data
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("./test_trace_" + str(prof.step_num) + ".json")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

    print(f"Utilizing GPU {torch.cuda.get_device_name(0)} for training and inference.")
else:
    print("Utiilzing CPU for training and inference.")

if False:
    seq_len = 8
    batch_size = 7
else:
    seq_len = 1
    batch_size = 1

training_data = frame_data.FrameData("C:\\FalcorFiles\\Dataset0\\", device, seq_len)
training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

model = fpcnn.UNetDLF2()
model = torch.nn.DataParallel(model)
model = model.to(device)
#model = torch.compile(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
loss_fn = torch.nn.L1Loss()

numIters = 3500
lossHistory = []
for i in range(0, numIters):
    input, target = next(iter(training_loader))

    optimizer.zero_grad()

    output = model(input)

    loss = loss_fn(output, target)
    loss.backward()

    optimizer.step()
    scheduler.step()

    print(f"Loss at iteration {i}\tis {loss.item()}")
    lossHistory.append(loss.item())

    if i == 0 or i == (numIters - 1):
        # first, export our model
        if i == (numIters - 1):
            torch.save(model.state_dict(), "C:/FalcorFiles/Models/FPCNN-3.pt")

        # get last few frames when it has stabilized
        image = output.detach()
        image = image[0].squeeze().permute((1, 2, 0)).cpu().numpy()
        image = image[:, :, -3:]

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

with torch.no_grad():
    input, target = training_data.get_full_img()
    input = input[None, :]
    target = target[None, :]

    model = model.to(device)
    output = model(input)

    loss = loss_fn(output, target)
    print(f"Loss on entire image was {loss.item()}")


    image = output.detach()
    image = image[0].squeeze().permute((1, 2, 0)).cpu().numpy()
    image = image[:, :, -3:]

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

plt.plot(lossHistory)
plt.show()
