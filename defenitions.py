import math
import numpy as np
import torch 
import torch.nn.functional as F
import torch.nn as nn


w_out, h_out = (128, 128)
#целевые точки для стандартного лица
dst_points = np.float32([
    [w_out * 0.35, h_out * 0.30],  # левый глаз
    [w_out * 0.65, h_out * 0.30],  # правый глаз
    [w_out * 0.50, h_out * 0.55],  # нос
    [w_out * 0.35, h_out * 0.70],  # левый угол рта
    [w_out * 0.65, h_out * 0.70]   # правый угол рта
])
  

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = scale
        self.m = margin
        
        self.W = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.W)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        loss = F.cross_entropy(output, labels)
        return loss