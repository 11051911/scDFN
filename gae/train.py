import torch
from opt import args
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []


def Pretrain_gae(model, data, adj, label, gamma_value):

    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):

        # if (args.name in use_adjust_lr):
        #     adjust_learning_rate(optimizer, epoch)

        z_igae, z_hat, adj_hat = model(data, adj)

        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss = loss_w + gamma_value * loss_a
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_igae.data.cpu().numpy())

        nmi, ari = eva(label, kmeans.labels_, epoch)

        nmi_result.append(nmi)
        ari_result.append(ari)

        torch.save(model.state_dict(), args.model_save_path)

