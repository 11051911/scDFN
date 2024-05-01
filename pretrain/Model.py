import torch
from AE import AE
from GAE import IGAE
from opt import args
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from ZINB import decoder_ZINB

class Pre_model(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,gae_n_enc_4,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,gae_n_dec_4,
                 n_input, n_z, n_clusters, layerd, hidden, dropout,n4=100,v=1.0, n_node=None, device=None):
        super(Pre_model, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.ae.load_state_dict(torch.load(args.ae_model_save_path))

        self.gae = IGAE(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_enc_4=gae_n_enc_4,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            gae_n_dec_4=gae_n_dec_4,
            n_input=n_input)
        self.Decoder = decoder_ZINB(layerd, hidden, n4, dropout)

        self.gae.load_state_dict(torch.load(args.gae_model_save_path))

        self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True).to(device)
        self.b = 1 - self.a

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        pi, disp, mean = self.Decoder(z_tilde)

        self.mean = mean
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        return x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, pi, disp, mean
