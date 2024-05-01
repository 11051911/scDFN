import opt
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils import adjust_learning_rate
from utils import eva, target_distribution
from ZINB import ZINB
from tensorboardX import SummaryWriter, writer

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []

writer = SummaryWriter('logs1')

def Train(epoch, model, data, adj, label, lr, pre_model_save_path, final_model_save_path, n_clusters,
          original_acc, gamma_value, lambda_value, device):
    optimizer = Adam(model.parameters(), lr=lr)
    model.load_state_dict(torch.load(pre_model_save_path, map_location='cpu'))  #_pretrain.pkl
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde, _, _, _ = model(data, adj)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(label, cluster_id, 'Initialization')

    for epoch in range(epoch):
        # if opt.args.name in use_adjust_lr:
        #     adjust_learning_rate(optimizer, epoch)
        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde, pi, disp, mean  = model(data, adj)

        tmp_q = q.data
        p = target_distribution(tmp_q)
        zinb = ZINB(pi, theta=disp, ridge_lambda=0)
        loss_zinb = zinb.loss(data, mean, mean=True)
        loss_zinb = loss_zinb.double()
        loss_ae = F.mse_loss(x_hat, data)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_igae = loss_w + gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = 0.3*loss_ae + 0.3*loss_igae + 0.3*lambda_value * loss_kl + 0.1*loss_zinb

        if (epoch+1) % 10 == 0:
            print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

        nmi, ari = eva(label, kmeans.labels_, epoch)
        if (epoch+1) % 10 == 0:
            print('Epoch_{}'.format(epoch), 'nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              )

        nmi_result.append(nmi)
        ari_result.append(ari)

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('nmi', nmi, epoch)
        writer.add_scalar('ari', ari, epoch)
        torch.save(model.state_dict(), final_model_save_path)
    writer.close()
