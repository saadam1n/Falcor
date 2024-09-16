import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.query_mtx = nn.ParameterList()
        self.key_mtx = nn.ParameterList()
        self.value_mtx = nn.ParameterList()

        self.numLayers = 1
        for layer in range(self.numLayers):
            qm = nn.Linear(8, 8, bias=False)
            km = nn.Linear(8, 8, bias=False)
            vm = nn.Linear(8, 8, bias=False)

            qm.weight.data.copy_(torch.eye(8) + torch.rand(8, 8) * 0.01)
            km.weight.data.copy_(torch.eye(8) + torch.rand(8, 8) * 0.01)
            vm.weight.data.copy_(torch.eye(8) + torch.rand(8, 8) * 0.01)

            self.query_mtx.append(qm)
            self.key_mtx.append(km)
            self.value_mtx.append(vm)

    def generateNormWeights(self, prod):
        if False:
            prod = prod / math.sqrt(8)
        elif False:
            prod = 8 * prod
            weights =  F.softmax(prod, dim = 1)
            #
            return weights
        else:
            prod = F.relu(prod)

            weights = F.normalize(prod, p=1.0, dim=1)

            return weights

    def forward(self, input):
        # first format the input tensor into a format we actually want
        embedding = torch.permute(input, (1, 2, 0)).view(25, 8)

        #print("This is the incoming embedding:")
        #print(embedding)

        for layer in range(self.numLayers):
            q = self.query_mtx[layer](embedding)
            k = self.key_mtx[layer](embedding)
            v = embedding#self.value_mtx[layer](embedding)

            k = k.transpose(0, 1)

            prod = torch.matmul(q, k)
            #print("This is product")
            #print(prod)



            weights = self.generateNormWeights(prod)#F.softmax(prod, dim=1)
            #weights = weights * weights
            #print("This is weights")
            #print(weights)

            ftform = torch.matmul(weights, v)

            alpha = 1.0
            embedding = ftform * (1.0 - alpha) + embedding * alpha

        return embedding[:, 0:4].view(5, 5, 4).permute((2, 0, 1))





