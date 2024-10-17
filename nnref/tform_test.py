import torch
import torch.nn as nn
import torch.nn.functional as F
import math

arrm = [
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
        [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
]

tformarr = list()

for i in range(3):
    layerlist = list()
    for j in range(4):
        tlist = list()
        for k in range(4):
            val = (16.0 * i) + (4.0 * j) + k
            sv = 2.0 + math.sin(val)
            tlist.append(sv)

        layerlist.append(tlist)
    tformarr.append(layerlist)

tformQ = torch.Tensor(tformarr[0]).transpose(0, 1)
tformK = torch.Tensor(tformarr[1]).transpose(0, 1)
tformV = torch.Tensor(tformarr[2]).transpose(0, 1)

mtx = torch.Tensor(arrm).permute((1, 2, 0))
#print(mtx)
mtx = mtx.view(25, 4)
#print(mtx)

q = torch.matmul(mtx, tformQ)
k = torch.matmul(mtx, tformK).transpose(0, 1)
v = torch.matmul(mtx, tformV)

scores = torch.matmul(q, k)
scores = F.softmax(scores, dim=1)

finalmtx = torch.matmul(scores, v).permute((1, 0)).view(4, 5, 5)


print(finalmtx)
#print(q.permute((1, 0)).view(4, 5, 5))
#print(scores)
#print(v.permute((1, 0)).view(4, 5, 5))
