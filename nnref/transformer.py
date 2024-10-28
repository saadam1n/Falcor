import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, blocksize):
        super().__init__()

        self.blocksize = blocksize

        self.query_mtx = nn.ParameterList()
        self.key_mtx = nn.ParameterList()
        self.value_mtx = nn.ParameterList()
        self.preproc = nn.ParameterList()
        self.postproc = nn.ParameterList()

        self.numLayers = 8
        for layer in range(self.numLayers):
            qm = nn.Linear(8, 8, bias=False)
            km = nn.Linear(8, 8, bias=False)
            vm = nn.Linear(8, 8, bias=False)

            qm.weight.data.copy_(torch.eye(8) * math.pow(8.0, 1.0 / 8.0) + torch.rand(8, 8) * 0.01)
            km.weight.data.copy_(torch.eye(8) + torch.rand(8, 8) * 0.01)
            vm.weight.data.copy_(torch.eye(8) + torch.rand(8, 8) * 0.01)

            self.query_mtx.append(qm)
            self.key_mtx.append(km)
            self.value_mtx.append(vm)

            prel = nn.Linear(8, 8)
            postl = nn.Linear(8, 8)

            self.preproc.append(prel)
            self.postproc.append(postl)

    def generateNormWeights(self, prod):
        if True:
            prod = prod / math.sqrt(8)
            weights = F.softmax(prod, dim = 1)
            return weights
        elif True:
            prod = 8 * prod
            weights =  F.softmax(prod, dim = 1)
            return weights
        else:
            prod = F.relu(prod)

            weights = F.normalize(prod, p=1.0, dim=1)

            return weights

    def forward(self, input):
        # first format the input tensor into a format we actually want
        embedding = torch.permute(input, (1, 2, 0)).view(self.blocksize * self.blocksize, 8)

        #print("This is the incoming embedding:")
        #print(embedding)

        for layer in range(self.numLayers):
            # for each feature, normalize it
            (emba, embv) = torch.var_mean(embedding, dim=0)
            emba = emba.view(1, 8).expand(self.blocksize * self.blocksize, 8)
            embv = embv.view(1, 8).expand(self.blocksize * self.blocksize, 8)
            embnorm = (embedding - emba) / embv

            embnorm = torch.nan_to_num(embnorm)

            q = self.query_mtx[layer](embnorm)
            k = self.key_mtx[layer](embnorm)
            v = self.value_mtx[layer](embedding)

            k = k.transpose(0, 1)
            prod = torch.matmul(q, k)

            #lq = torch.matmul(q, q.transpose(0, 1))
            #lk = torch.matmul(k.transpose(0, 1), k).transpose(0, 1)

            #print(lq.size())
            #print(lk.size())

            #prod = 2 * prod - lq - lk

            #print("This is product")
            #print(prod)



            weights = self.generateNormWeights(prod)#F.softmax(prod, dim=1)
            #weights = weights * weights
            #print("This is weights")
            #print(weights)

            ftform = torch.matmul(weights, v)

            ftform = self.preproc[layer](ftform)
            ftform = F.relu(ftform)
            ftform = self.postproc[layer](ftform)

            alpha = 0.8
            embedding = ftform * (1.0 - alpha) + embedding * alpha

        num_out_channels = 3
        return embedding[:, 0:num_out_channels].view(self.blocksize, self.blocksize, num_out_channels).permute((2, 0, 1))





