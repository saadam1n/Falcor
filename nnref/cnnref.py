import torch
import torch.nn as nn
import torch.nn.functional as F

print("Hello World, this language is very unintuitive")

class KpcnnTest(nn.Module):
    def __init__(self):
        super(KpcnnTest, self).__init__()

        self.numLayers = 4
        self.numWeights = 8

        self.conv = list()
        for layer in range(self.numLayers):
            inchnl = 8
            outchnl =8

            if(layer == 0):
                inchnl = 8
            elif(layer == self.numLayers - 1):
                outchnl = 8

            curconv = nn.Conv2d(inchnl, outchnl, 3, padding=1)
            self.conv.append(curconv)

        # 4 illum channels
        self.postconv = nn.Conv2d(8, 4 * self.numWeights, 5)

        self.smax = nn.Softmax(dim=0)
        self.kmax = nn.Softmax2d()
    def forward(self, input):
        # apply convolution
        convmaps = input

        for layer in range(self.numLayers):
            convmaps = F.relu(self.conv[layer](convmaps))



        #normPostconv = self.kmax(self.postconv.weight.data)
        postconvcolors = self.postconv(input).view(4, self.numWeights)
        normWeights = self.smax(convmaps)
        filtered = torch.Tensor(4, 5, 5)

        for y in range(5):
            for x in range(5):
                filtered[:, y, x] = torch.matmul(postconvcolors, normWeights[:, y, x])

        return filtered

kpcnntest = KpcnnTest()

rawdata = [
    [
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 1.0, 1.0, 0.0, 0.0,],
        [0.0, 1.0, 1.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, -1.0, -1.0, 0.0,],
        [0.0, 0.0, -1.0, -1.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ],
]

# two more illum channels, then 4 channels more for normal + depth
for blankmap in range(0, 6):
    rawdata.append([
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ])

targetdata = list()
for channel in range(4):
    targetdata.append(rawdata[channel])

inputtensor = torch.tensor(rawdata)
targettensor = torch.tensor(targetdata)

#o = cnntest.forward(t)
#print(o.size())

crit = nn.L1Loss()
optim = torch.optim.SGD(kpcnntest.parameters(), lr=0.001, momentum=0.9)

for epoch in range(200):
    optim.zero_grad()

    outputtensor = kpcnntest(inputtensor)
    loss = crit(outputtensor, targettensor)
    loss.backward()
    optim.step()

    print(loss.item())

    if(epoch == 0 or epoch == 199):
        print(outputtensor)


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




