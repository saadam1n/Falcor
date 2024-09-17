import torch
import torch.nn as nn
import torch.nn.functional as F
import kpcnn
import transformer
import miniblock_transformer
import math
import numpy
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


import cv2

print("Hello World, this language is very unintuitive")
torch.set_printoptions(precision=4, sci_mode=False)

reference = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Reference.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
illumination = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Color.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
albedo = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Albedo.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
emission = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-Emission.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
normals = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-WorldNormal.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
linearz = cv2.imread("C:\\FalcorFiles\\Dataset0\\0-LinearZ.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

startx = 620
starty = 620

refPatch = reference[starty:starty + 5, startx:startx + 5, 0:3]
illumPatch = illumination[starty:starty + 5, startx:startx + 5, 0:3]
normalPatch = normals[starty:starty + 5, startx:startx + 5, 0:3]
linearZPatch = linearz[starty:starty + 5, startx:startx + 5, 2:3]

# basically we need to set up our input data
# illum + 0 + normalPatch + linearZ

refTensor = torch.Tensor(refPatch)
illumTensor = torch.tensor(illumPatch)
normalTensor = torch.tensor(normalPatch)
linearZTensor = torch.tensor(linearZPatch)

inputtensor = torch.cat((illumTensor, torch.zeros(5, 5, 1), normalTensor, linearZTensor), dim=2).permute(2, 0, 1)
targettensor = torch.cat((refTensor, torch.zeros(5, 5, 1)), dim=2).permute(2, 0, 1)

print(inputtensor.shape)
print(targettensor.shape)



model = transformer.Transformer()

#o = cnntest.forward(t)
#print(o.size())

count_dict = dict()

for name, param in model.named_parameters():
    #if param.requires_grad:
        #print(name, param.data)
    count_dict[name] = 0


crit = nn.L1Loss()
optim = torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.9)

num_training_iter = 1000
for epoch in range(num_training_iter):
    optim.zero_grad()

    outputtensor = model(inputtensor)
    loss = crit(outputtensor, targettensor)
    loss.backward()
    optim.step()

    #for name, param in model.named_parameters():
    #    if 'weight' in name:
    #        temp = torch.zeros(param.grad.shape)
    #        temp[param.grad != 0] += 1
    #        count_dict[name] += temp

    print(loss.item() * 100.0)

    if(epoch == 0 or epoch == num_training_iter - 1):
        print(outputtensor)

#print(count_dict)


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




