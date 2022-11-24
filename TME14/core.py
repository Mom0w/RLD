import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#GLOW : ActNorm, AffineCouplingLayer, Convolution 1x1

class AffineFlow(nn.Module):

    def __init__(self, shape):
        
        super().__init__()
 
        self.s = nn.Parameter(torch.zeros(shape)[None])
        self.register_buffer('s', torch.zeros(shape)[None])
        
        self.t = nn.Parameter(torch.zeros(shape)[None])
        self.register_buffer('t', torch.zeros(shape)[None])
        
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det

class ActNorm(AffineFlow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_done = torch.tensor(0.)
        self.register_buffer('init_done', self.init_done)

    def forward(self, z):
        if not self.init_done:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)).data
            self.init_done = torch.tensor(1.)
        return super().forward(z)

    def inverse(self, z):
        
        if not self.init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.init_done = torch.tensor(1.)
        return super().inverse(z)


class AffineCouplingLayer(nn.Module):
    def __init__(self,inSize,outSize,latSize):
        super().__inti__()
        self.net = MLP(inSize,outSize,latSize)
        
    def forward(self,z):
        """
        Input 
        z : [z1,z2] Moitie des valeurs 
        
        Output
        y : [y1,y2]
        log_det: log determinant 
        """
        z1, z2 = z 
        features = self.net(z1)
        shift = features[:, 0::2, ...]
        scale = features[:, 1::2, ...]
        y1 = z1
        y2 = z2 * torch.exp(scale) + shift
        log_det = torch.sum(scale, dim=list(range(1, shift.dim())))
        
        return [y1,y2], log_det

    def inverse(self,z):
        """
        Input 
        z : [z1,z2] Moitie des valeurs 
        
        Output
        y : [y1,y2]
        log_det: log determinant 
        """
        z1, z2 = z 
        features = self.net(z1)
        shift = features[:, 0::2, ...]
        scale = features[:, 1::2, ...]
        y1 = z1
        y2 = (z2 - shift) * torch.exp(-scale)
        log_det = -torch.sum(scale, dim=list(range(1, shift.dim())))
        
        return [y1,y2], log_det
        
    
class Convolution1x1(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.channel = channel
        
        W = nn.init.orthogonal_(torch.empty(self.channel,self.channel))
        P, L, U = torch.lu_unpack(*W.lu())
        
        self.P = P # matrice de passage à ne pas optimiser
        self.L_ = nn.Parameter(torch.tril(L, diagonal=-1))
        self.U_ = nn.Parameter(torch.triu(U, diagonal=1))
        self.S = nn.Parameter(torch.diag(U))
        
        
    def forward(self,z):
        L = self.L_ + torch.eye(self.channel)
        U = self.U_ + self.S
        W = self.P @ L @ U
        
        log_det = -torch.log(torch.abs(self.S)).sum()
        return z @ W, log_det 
        
    
    def inverse(self,z):
        
        U_inv = torch.inverse(self.U_ + self.S)
        L_inv = torch.inverse(self.L_ + torch.eye(self.channel))
        W_inv = U_inv @ L_inv @ self.P.T
        
        log_det = torch.log(torch.abs(self.S)).sum()
        
        return z @ W_inv, log_det 
    
    
if __name__ == '__main__':
    
    #Conv 1x1 
    conv = Convolution1x1(10)
    z = torch.rand((32,10))
    x,log = conv(z)
    
    z_,log_ = conv.inverse(x)
    
    print(torch.dist(z, z_))
    
    
    
    
    
    
