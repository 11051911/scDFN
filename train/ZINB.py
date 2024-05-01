import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# import dgl
from torch_geometric.nn import TAGConv
sp=nn.Softplus()
MeanAct = lambda x : torch.clamp(torch.exp(x), 1e-5, 1e6)
DispAct = lambda x : torch.clamp(sp(x), 1e-4, 1e4)

class layer_zinb(nn.Module):
    def __init__(self, in_feature, out_feature) :
        super(layer_zinb, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.w=nn.Parameter(torch.zeros(size=(in_feature, out_feature)))
        nn.init.xavier_uniform_(self.w.data, gain=1)

    def forward(self, input, type):
        h=torch.mm(input, self.w)

        # activation
        if type=='sigmoid':
            layer=nn.Sigmoid()
            h=layer(h)
            #h=nn.Sigmoid(h)
        elif type=='MeanAct':
            #h=torch.exp(h)
            h=MeanAct(h)
        elif type=='DispAct':
            #layer=nn.Softplus()
            h=DispAct(h)

        return h


class Encoder(nn.Module):
    def __init__(self, n1, n2, n3, n4, dropout, activation=nn.ReLU(),g=None, n_clusters=None):
        super(Encoder, self).__init__()
        self.g=g
        self.dropout = nn.Dropout(p=dropout)

        if self.g!=None:
            self.enc1 = TAGConv(in_channels=n4, out_channels=n3)
            self.enc2 = TAGConv(in_channels=n3, out_channels=n2)
            self.enc3 = TAGConv(in_channels=n2, out_channels=n1)
            #self.enc4 = TAGConv(in_channels=n1, out_channels=n_clusters)

        else:
            self.enc1 = nn.Linear(n4, n3)
            self.BN1 = nn.BatchNorm1d(n3, momentum=0.01, eps=0.001)
            self.enc2 = nn.Linear(n3, n2)
            self.BN2 = nn.BatchNorm1d(n2, momentum=0.01, eps=0.001)
            self.enc3 = nn.Linear(n2, n1)
            self.BN3 = nn.BatchNorm1d(n1, momentum=0.01, eps=0.001)
            #self.enc4 = nn.Linear(n1, n_clusters)

    def forward_g1(self, x):
      if self.g!=None:
        enc_h1 = self.dropout(self.enc1(x, self.g))
        return enc_h1
    def forward_g2(self, x):
      if self.g!=None:
        enc_h2 = self.dropout(self.enc2(x, self.g))
        return enc_h2
    def forward_g3(self, x):
      if self.g!=None:
        enc_h3 = self.dropout(self.enc3(x, self.g))
        return enc_h3
    
    def forward(self, x): # ?

        if self.g!=None:
            enc_h1 = self.dropout(self.enc1(x, self.g))
            enc_h2 = self.dropout(self.enc2(enc_h1, self.g))
            enc_h3 = self.dropout(self.enc3(enc_h2, self.g))

            h=F.normalize(enc_h3, p=2, dim=1)
            
        else:
            enc_h1 = self.dropout(nn.ReLU(self.BN1(self.enc1(x))))
            enc_h2 = self.dropout(nn.ReLU(self.BN2(self.enc2(enc_h1))))
            enc_h3 = self.dropout(nn.ReLU(self.BN3(self.enc3(enc_h2))))
            h = enc_h3

        
        return h, enc_h1, enc_h2, enc_h3

    def dot_product_decoder(self, Z):
        A_pred=torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

class decoder_ZINB(nn.Module):
    def __init__(self, n_layers, hidden, input_size, dropout) :
        super(decoder_ZINB, self).__init__()

        self.decoder=nn.ModuleList()
        #dec_dim?
        for i , (n_in, n_out)in enumerate(zip(n_layers[:-1], n_layers[1:])):
            self.decoder.append(nn.Linear(n_in, n_out))
            self.decoder.append(nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Dropout(p=dropout))
    
        self.layer_zinb=layer_zinb(n_layers[-1], input_size)# ?

    def forward(self, z):
        latent=z
        for i, layer in enumerate(self.decoder):
            latent=layer(latent.float())
        #latent=self.decoder(z)
        # pi
        pi=self.layer_zinb(latent, type='sigmoid')
        disp=self.layer_zinb(latent, type='DispAct')
        mean=self.layer_zinb(latent, type='MeanAct')

        return pi, disp, mean

def _nelem(x):
  isnan=~torch.isnan(x)
  isnan=isnan.type(torch.FloatTensor)
  nelem=torch.sum(isnan)

def _nan2zero(x):
  return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
  return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)

def _reduce_mean(x):
    # x = x.astype(np.)
    nelem=_nelem(x)
    x=_nan2zero(x)
    return torch.divide(torch.reduce_sum(x),nelem)

class NB(object):
  def __init__(self, theta=None, masking=False, scale_factor=1.0):
    self.eps=1e-10
    self.scale_factor=scale_factor
    self.masking=masking
    self.theta=theta
  
  def loss(self, y_true, y_pred, mean=True):
    scale_factor=self.scale_factor
    eps=self.eps
    if self.masking:
      nelem=_nelem(y_true)
      y_true=_nan2zero(y_true)
    
    a=torch.full(self.theta.size(), 1e6)
    a=a.to("cuda:0")
    theta=torch.minimum(self.theta, a)

    t1=torch.lgamma(theta+eps)+torch.lgamma(y_true+1.0)-torch.lgamma(y_true+theta+eps)
    t2=(theta+y_true)*torch.log(1.0+(y_pred/(theta+eps)))+(y_true*(torch.log(theta+eps)-torch.log(y_pred+eps)))
    final=t1+t2
    final=_nan2inf(final)

    if mean:
      if self.masking:
        final=torch.divide(torch.reduce_sum(final), nelem)
      else:
        final=torch.mean(final)
    return final

class ZINB(NB):
  def __init__(self, pi, ridge_lambda=0.0, **kwargs):
      super().__init__(**kwargs)
      self.pi=pi
      self.ridge_lambda=ridge_lambda

  def loss(self, y_true, y_pred, mean=True):
    scale_factor=self.scale_factor
    eps=self.eps
    
    nb_case=super().loss(y_true, y_pred, mean=False)-torch.log(1.0-self.pi)
    y_true=y_true.type(torch.FloatTensor)
    y_pred=y_pred.type(torch.FloatTensor)
    y_pred=y_pred*scale_factor
    a=torch.full(self.theta.size(), 1e6)
    a=a.to("cuda:0")
    theta=torch.minimum(self.theta, a)
    theta=theta.to("cuda:0")
    #eps=eps.to("cuda:0")
    eps=torch.full(y_pred.size(), self.eps)
    eps=eps.to("cuda:0")
    y_pred=y_pred.to("cuda:0")
    y_true=y_true.to("cuda:0")

    zero_nb=torch.pow(theta/(theta+y_pred+eps), theta)
    zero_case=-torch.log(self.pi+((1.0-self.pi)*zero_nb)+eps)
    result=torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
    ridge=self.ridge_lambda*torch.square(self.pi)
    result+=ridge

    if mean:
      if self.masking:
        result=_reduce_mean(result)
      else:
        result=torch.mean(result)
    
    result = _nan2inf(result)

    return result






