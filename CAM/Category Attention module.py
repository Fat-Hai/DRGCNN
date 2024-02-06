import torch.nn.functional as F
from torch import nn
import torch

class Category_attention_block(nn.Module):
    def __init__(self, dim,classes, k):
        super(Category_attention_block, self).__init__()
        self.classes = classes
        self.k = k
        self.conv = nn.Conv2d(in_channels=dim, out_channels=k*classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(k*classes)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        shape = inputs.shape

        C = self.conv(inputs)
        C = self.bn(C)
        F1 = self.relu(C)

        F2 = F1
        x = F.adaptive_max_pool2d(F2, (1, 1))

        shape_x = x.shape
        x = x.view(shape_x[0],shape_x[2],shape_x[3],self.classes,self.k)

        S = x.mean(dim=-1, keepdim=False)

        x = F1.view(shape[0], shape[2], shape[3], self.classes, self.k)

        x = x.mean(dim=-1, keepdim=False)

        x = torch.mul(x, S)

        M = x.mean(dim=-1, keepdim=True)

        M = M.view(shape_x[0],1,shape[2],shape[3])

        semantic = torch.mul(inputs , M)

        return semantic