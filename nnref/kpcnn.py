import torch
import torch.nn as nn
import torch.nn.functional as F

class Kpcnn(nn.Module):
    def __init__(self):
        super(Kpcnn, self).__init__()

        self.doposenc = False

        self.numLayers = 8
        self.numWeights = 6


        self.conv = nn.ParameterList()
        for layer in range(self.numLayers):
            inchnl = 25
            outchnl = 25

            if(layer == 0):
                if(self.doposenc):
                    inchnl = 10
                else:
                    inchnl = 8
            if(layer == self.numLayers - 1):
                outchnl = self.numWeights

            curconv = nn.Conv2d(inchnl, outchnl, 3, padding=1)
            self.conv.append(curconv)

        # 4 illum channels
        self.postconv = nn.Conv2d(3, 3 * self.numWeights, 5)

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
            convmaps = self.conv[layer](convmaps)
            if(layer != self.numLayers - 1):
                convmaps = F.relu(convmaps)



        postconvcolors = self.postconv(input[0:3]).view(3, self.numWeights)
        normWeights = F.softmax(convmaps, dim=0)
        filtered = torch.Tensor(3, 5, 5)

        for y in range(5):
            for x in range(5):
                filtered[:, y, x] = torch.matmul(postconvcolors, normWeights[:, y, x])
                #print(normWeights[:, y, x])

        return filtered
