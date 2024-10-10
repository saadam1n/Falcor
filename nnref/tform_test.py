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
]

mtx = torch.Tensor(arrm).permute((1, 2, 0))
print(mtx)
mtx = mtx.view(25, 2)
print(mtx)

scores = torch.matmul(mtx, mtx.transpose(0, 1))
scores = F.softmax(scores, dim=1)

finalmtx = torch.matmul(scores, mtx).permute((1, 0)).view(2, 5, 5)


print(finalmtx)
