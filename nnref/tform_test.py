import torch
import torch.nn as nn
import torch.nn.functional as F

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
for i in range(4):
    tlist = list()
    for j in range(4):
        val = (4.0 * i) + j
        tlist.append(val)

    tformarr.append(tlist)

tform = torch.Tensor(tformarr).transpose(0, 1)

mtx = torch.Tensor(arrm).permute((1, 2, 0))
#print(mtx)
mtx = mtx.view(25, 4)
#print(mtx)

q = torch.matmul(mtx, tform)
k = torch.matmul(mtx, tform).transpose(0, 1)
v = torch.matmul(mtx, tform)

scores = torch.matmul(q, k)
scores = F.softmax(scores, dim=1)

finalmtx = torch.matmul(scores, v).permute((1, 0)).view(4, 5, 5)


print(finalmtx)
print(q.permute((1, 0)).view(4, 5, 5))
print(scores)
print(v.permute((1, 0)).view(4, 5, 5))
