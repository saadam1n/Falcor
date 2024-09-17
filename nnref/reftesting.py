import torch
import torch.nn as nn
import torch.nn.functional as F
import kpcnn
import transformer
import miniblock_transformer
import math
import numpy
import os
import random
import matplotlib.pyplot as plt
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


import cv2

print("Hello World, this language is very unintuitive")
torch.set_printoptions(precision=4, sci_mode=False)

reference = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Reference.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
color = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Color.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
albedo = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Albedo.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
emission = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Emission.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
normals = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-WorldNormal.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
linearz = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-LinearZ.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

model = transformer.Transformer()

#o = cnntest.forward(t)
#print(o.size())

count_dict = dict()

for name, param in model.named_parameters():
    #if param.requires_grad:
        #print(name, param.data)
    count_dict[name] = 0


crit = nn.L1Loss()
optim = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9)

alphaloss = -1.0
losshistory = list()

num_training_iter = 5000
for epoch in range(num_training_iter):
    startx = random.randint(0, 1920 - 5 - 1)
    starty = random.randint(0, 1080 - 5 - 1)

    refPatch = reference[starty:starty + 5, startx:startx + 5, 0:3]
    colorPatch = color[starty:starty + 5, startx:startx + 5, 0:3]
    albedoPatch = albedo[starty:starty + 5, startx:startx + 5, 0:3]
    emissionPatch = emission[starty:starty + 5, startx:startx + 5, 0:3]
    normalPatch = normals[starty:starty + 5, startx:startx + 5, 0:3]
    linearZPatch = linearz[starty:starty + 5, startx:startx + 5, 2:3]

#Average loss was: 38.31475299393369 with squared diff
#Average loss was: 30.6063318612274 without it
    # basically we need to set up our input data
    # illum + 0 + normalPatch + linearZ



    refTensor = torch.Tensor(refPatch)
    colorTensor = torch.tensor(colorPatch)
    albedoTensor = torch.tensor(albedoPatch)
    emissionTensor = torch.tensor(emissionPatch)
    normalTensor = torch.tensor(normalPatch)
    linearZTensor = torch.tensor(linearZPatch)

    albedoMask = torch.lt(albedoTensor, 0.01)
    albedoTensor[albedoMask] = 1.0

    illumTensor = (colorTensor - emissionTensor) / albedoTensor

    inputtensor = torch.cat((illumTensor, torch.zeros(5, 5, 1), normalTensor, linearZTensor), dim=2).permute(2, 0, 1)
    targettensor = refTensor.permute(2, 0, 1)


    optim.zero_grad()

    outputtensor = model(inputtensor) * albedoTensor.permute(2, 0, 1) + emissionTensor.permute(2, 0, 1)
    loss = crit(outputtensor, targettensor)
    loss.backward()
    optim.step()

    #for name, param in model.named_parameters():
    #    if 'weight' in name:
    #        temp = torch.zeros(param.grad.shape)
    #        temp[param.grad != 0] += 1
    #        count_dict[name] += temp

    if(alphaloss == -1):
        alphaloss = loss.item()
    else:
        mvingavg = 0.8
        alphaloss =alphaloss * mvingavg + loss.item() * (1.0 - mvingavg)

    print("Epoch {}: loss is {}".format(epoch, alphaloss * 100.0))
    losshistory.append(loss.item())

    if(epoch == 0 or epoch == num_training_iter - 1):
        print(outputtensor)

#print(count_dict)
print("Average loss was: {}".format(100.0 * numpy.average(losshistory[int(0.8*num_training_iter):-1])))
plt.plot(losshistory)
plt.show()

if False:
    for y in range(5):
        for x in range(5):
            ppRWList = list()


            for w in range(8):
                ppRWList.append(o[w][y][x].item())
                #print(o[w][y][x].item(), end="\t")

            ppRW = torch.tensor(ppRWList);

            if True:
                ppW = nn.Softmax()(ppRW)
            else:
                ppW = ppRW

            print(ppW)




