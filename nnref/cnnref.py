import torch
import torch.nn as nn
import torch.nn.functional as F

print("Hello World, this language is very unintuitive")
torch.set_printoptions(precision=4, sci_mode=False)

class KpcnnTest(nn.Module):
    def __init__(self):
        super(KpcnnTest, self).__init__()

        self.doposenc = False

        self.numLayers = 1
        self.numWeights = 12


        self.conv = nn.ParameterList()
        for layer in range(self.numLayers):
            inchnl = 25
            outchnl = 25

            if(layer == 0):
                if(self.doposenc):
                    inchnl = 6
                else:
                    inchnl = 4
            if(layer == self.numLayers - 1):
                outchnl = self.numWeights

            curconv = nn.Conv2d(inchnl, outchnl, 3, padding=1)
            self.conv.append(curconv)

        # 4 illum channels
        self.postconv = nn.Conv2d(4, 4 * self.numWeights, 5)

        self.smax = nn.Softmax(dim=0)
    def forward(self, input):

        convmaps = input


        if(self.doposenc):
            posenc = list()

            for blankmap in range(0, 2):
                posenc.append([
                    [0.0, 0.0, 0.0, 0.0, 0.0,],
                    [0.0, 0.0, 0.0, 0.0, 0.0,],
                    [0.0, 0.0, 0.0, 0.0, 0.0,],
                    [0.0, 0.0, 0.0, 0.0, 0.0,],
                    [0.0, 0.0, 0.0, 0.0, 0.0,],
                ])

            for y in range(5):
                for x in range(5):
                    posenc[0][y][x] = y - 2
                    posenc[1][y][x] = x - 2

            convmaps = torch.cat((convmaps, torch.Tensor(posenc)), 0)

        # apply convolution
        for layer in range(self.numLayers):
            convmaps = F.relu(self.conv[layer](convmaps))



        postconvcolors = self.postconv(input).view(4, self.numWeights)
        normWeights = convmaps#F.softmax(convmaps, dim=0)
        filtered = torch.Tensor(4, 5, 5)

        for y in range(5):
            for x in range(5):
                filtered[:, y, x] = torch.matmul(postconvcolors, normWeights[:, y, x])
                #print(normWeights[:, y, x])

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
for blankmap in range(0, 2):
    rawdata.append([
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ])



print(rawdata)

targetdata = list()
for channel in range(4):
    targetdata.append(rawdata[channel])

inputtensor = torch.tensor(rawdata)
targettensor = torch.tensor(targetdata)

#o = cnntest.forward(t)
#print(o.size())

count_dict = dict()

for name, param in kpcnntest.named_parameters():
    #if param.requires_grad:
        #print(name, param.data)
    count_dict[name] = 0


crit = nn.L1Loss()
optim = torch.optim.SGD(kpcnntest.parameters(), lr=0.001, momentum=0.9)

num_training_iter = 1000
for epoch in range(num_training_iter):
    optim.zero_grad()

    outputtensor = kpcnntest(inputtensor)
    loss = crit(outputtensor, targettensor)
    loss.backward()
    optim.step()

    for name, param in kpcnntest.named_parameters():
        if 'weight' in name:
            temp = torch.zeros(param.grad.shape)
            temp[param.grad != 0] += 1
            count_dict[name] += temp

    print(loss.item())

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




