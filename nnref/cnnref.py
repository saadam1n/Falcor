import torch
import torch.nn as nn
import torch.nn.functional as F

print("Hello World, this language is very unintuitive")

class CnnTest(nn.Module):
    def __init__(self):
        super(CnnTest, self).__init__()

        self.numLayers = 4

        self.conv = list()
        for layer in range(self.numLayers):
            curconv = nn.Conv2d(8, 8, 3, padding=1)

            rawkernel = list()
            for och in range(8):
                singlechannel = list()
                for ich in range(8):
                    singlechannel.append([
                        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
                        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
                        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
                    ])
                rawkernel.append(singlechannel)

            with torch.no_grad():
                curconv.weight.data = torch.tensor(rawkernel)

            self.conv.append(curconv)


    def forward(self, input):
        # apply convolution
        convmaps = input

        if(False):
            for layer in range(self.numLayers):
                print(self.conv[layer].weight.data[7][7][2][2].item())
                convmaps = F.relu(self.conv[layer](convmaps))
        else:
            for layer in range(self.numLayers):
                curconv = nn.Conv2d(8, 8, 3, padding=1)

                rawkernel = list()
                for och in range(8):
                    singlechannel = list()
                    for ich in range(8):
                        singlechannel.append([
                            [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
                            [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
                            [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
                        ])
                    rawkernel.append(singlechannel)

                convmaps = F.conv2d(convmaps, torch.tensor(rawkernel), padding=1)

        return convmaps

cnntest = CnnTest()


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

for blankmap in range(0, 6):
    rawdata.append([
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0,],
    ])

t = torch.tensor(rawdata)

o = cnntest.forward(t)

print(o.size())

for y in range(5):
    for x in range(5):
        ppRWList = list()


        for w in range(8):
            ppRWList.append(o[w][y][x].item())
            #print(o[w][y][x].item(), end="\t")

        ppRW = torch.tensor(ppRWList);

        ppW = nn.Softmax()(ppRW)

        print(ppW)




