import torch
import numpy as np
import simple_kernel
import kpcnn
import fpcnn
import frame_data
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import portable_keyboard
from pynput import keyboard
import time
import datetime
from noisebase import Noisebase
import platform

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

        print(f"Utilizing GPU {torch.cuda.get_device_name(0)} for training and inference.")
    else:
        print("Utiilzing CPU for training and inference.")

    if False:
        seq_len = 8
        batch_size = 4
    else:
        seq_len = 1
        batch_size = 8

    if True:
        #                                                                            /media/saad/00486DF1486DE5BE/FalcorFiles/Dataset0/
        path = "C:\\FalcorFiles\\Dataset0\\" if platform.system() == "Windows" else "/media/saad/00486DF1486DE5BE/FalcorFiles/Dataset0/"

        training_data = frame_data.FrameData(path, device, seq_len)
        training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    else:
        training_loader = Noisebase(
            'sampleset_v1',
            {
                'framework': 'torch',
                'buffers': ['color', 'diffuse', 'position', 'normal', 'reference'],
                'samples': 8,
                'batch_size': 16
            }
        )

    model = fpcnn.KernelRegressionDenoiser()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    #model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    loss_fn = torch.nn.L1Loss()

    def display_full_image(model, training_data, loss_fn):
        with torch.no_grad():
            input, target = training_data.get_full_img()
            input = input[None, :]
            target = target[None, :]

            model = model.to(device)
            output = model(input)

            loss = loss_fn(output, target)
            print(f"Loss on entire image was {loss.item()}")


            image = output.detach().pow(1 / 2.2)
            image = image[0].squeeze().permute((1, 2, 0)).cpu().numpy()
            image = image[:, :, -3:]

            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    model.train()


    numIters = 5000
    lossHistory = []
    for i in range(0, numIters):

        start = time.time()
        input, target = next(iter(training_loader))

        optimizer.zero_grad()

        output = model(input)

        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()
        scheduler.step()


        total_loss = loss.item()
        end = time.time()

        eta = (end - start) * (numIters - i)

        duration = str(datetime.timedelta(seconds=int(eta)))

        print(f"ETA [{duration}]:\tLoss at iteration {i}\tis {total_loss}")
        lossHistory.append(total_loss)

        if i == 0 or i == (numIters - 1) or portable_keyboard.is_key_pressed(keyboard.Key.f10):
            # first, export our model
            #if i == (numIters - 1):
                #torch.save(model.state_dict(), "C:/FalcorFiles/Models/IFWE.pt") # global pre-context filter

            # get last few frames when it has stabilized
            image = output.detach()
            image = image[0].squeeze().permute((1, 2, 0)).cpu().numpy()
            image = image[:, :, -3:]

            first_iter = True
            while portable_keyboard.is_key_pressed(keyboard.Key.f10):
                if first_iter:
                    print("Please release the key to display the window.")
                    first_iter = False


            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



        if portable_keyboard.is_key_pressed(keyboard.Key.f9):
            model.eval()

            first_iter = True
            while portable_keyboard.is_key_pressed(keyboard.Key.f9):
                if first_iter:
                    print("Please release the key to display the window.")
                    first_iter = False

            display_full_image(model, training_data, loss_fn)

            model.train()


    display_full_image(model, training_data, loss_fn)

    plt.plot(lossHistory)
    plt.show()
