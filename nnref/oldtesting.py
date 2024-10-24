import torch
import torch.nn as nn
import torch.nn.functional as F
import kpcnn
import transformer
import miniblock_transformer
import math

print("Hello World, this language is very unintuitive")
torch.set_printoptions(precision=4, sci_mode=False)

model = transformer.Transformer(5)

rawdata = [
    [
        [1.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 1.0, 1.0, 0.0, 0.0,],
        [0.0, 1.0, 1.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 1.0, 1.0, 0.0,],
        [0.0, 0.0, 1.0, 1.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ],
]

# two more illum channels, then 4 channels more for normal + depth
for blankmap in range(0, 0):
    rawdata.append([
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ])

normz = torch.randn(4, 5, 5) * 0.0

print(rawdata)

targetdata = list()
for channel in range(4):
    targetdata.append(rawdata[channel])

targettensor = torch.tensor(targetdata)
inputtensor = torch.tensor(rawdata)

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

    outputtensor = model(torch.cat((inputtensor + torch.abs(torch.randn(4, 5, 5)) * 0.0, normz), dim=0))
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




