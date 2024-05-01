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


def Pretrain_ae(model, dataset, y, train_loader, device):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        # if (args.name in use_adjust_lr):
        #     adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_hat, z = model(x)
            loss = F.mse_loss(x_hat, x)
            print('{} loss: {}'.format(epoch, loss))

            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())

            nmi, ari = eva(y, kmeans.labels_, epoch)

            nmi_result.append(nmi)
            ari_result.append(ari)


            torch.save(model.state_dict(), args.model_save_path)
