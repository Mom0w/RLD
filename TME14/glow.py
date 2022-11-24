import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GlowModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.GenerativeFlow  = nn.ModuleList([])
        
    def forward(self, z):
        log_det_sum = torch.zeros(z.shape[0], dtype=z.dtype, device=device)
        for flow in self.GenerativeFlow:
            z, log_det = flow(z)
            log_det_sum += log_det
        return z, log_det_sum

    def inverse(self, z):
        log_det_sum = torch.zeros(z.shape[0], dtype=z.dtype, device=device)
        for i in range(len(self.GenerativeFlow) - 1, -1, -1):
            z, log_det = self.GenerativeFlow[i].inverse(z)
            log_det_sum += log_det
        return z, log_det_sum
    

