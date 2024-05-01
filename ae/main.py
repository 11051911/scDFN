import torch
from AE import AE
import numpy as np
from opt import args
from utils import setup_seed
from torch.utils.data import Dataset, DataLoader
from train import Pretrain_ae
import pandas as pd
import scanpy as sc
from preprocess import prepro, normalize
from utils import get_adj
from scipy.sparse import coo_matrix
import h5py
import opt
setup_seed(1)

print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")



class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

# dataset_list1=[
#     "PBMC", "klein","kidney","romanov","Human1", "Human2", "Human3","Human4", "Mouse1", "Mouse2", "Zeisel", "HumanLiver"]
# dataset_list2=[
#     "Adam", "Chen","Muraro", "Pollen", "Quake_10x_Limb_Muscle",  "Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Heart", "Quake_Smart-seq2_Limb_Muscle",
#    "Quake_Smart-seq2_Lung","Wang_Lung",]
# dataset_list3=["Yan","Camp_Brain","Camp_Liver","Baron","biase","goolam","Human","Mouse","Xin","Tasic"]


args.model_save_path = 'model/model_save_ae/{}_ae.pkl'.format(args.name)

# data_mat = h5py.File('../dataset/{}.h5'.format(args.name))
# x = np.array(data_mat['X'])
# y = np.array(data_mat['Y'])
# data_mat.close()

#  x, y = prepro('../dataset2/{}/data.h5'.format(args.name))

x = np.array(pd.read_csv('../dataset3/{}/count.csv'.format(args.name), header=None))
y = np.array(pd.read_csv('../dataset3/{}/label.csv'.format(args.name), header=None))
if opt.args.name == "klein" or opt.args.name == "romanov" or opt.args.name == "Baron" or opt.args.name == "biase" or opt.args.name == "goolam" or opt.args.name == "Xin":
    x = x.T
else:
    x = x


x = np.ceil(x).astype(np.float32)
y = y.astype(np.float32)

cluster_number = int(max(y) - min(y) + 1)
print(cluster_number)
args.n_clusters = cluster_number
adata = sc.AnnData(x)
adata = normalize(adata, filter_min_counts=True, highly_genes=2000, size_factors=True,
                  normalize_input=False, logtrans_input=True)
print(adata)

Nsample1, Nfeature = np.shape(adata.X)
y = y.reshape(Nsample1, )
adj1, adjn1 = get_adj(adata.X)
A1 = coo_matrix(adj1)
X = torch.from_numpy(adata.X)
adjn1 = torch.from_numpy(adjn1)

dataset = LoadDataset(X)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
args.n_input = Nfeature

model = AE(
    ae_n_enc_1=args.ae_n_enc_1,
    ae_n_enc_2=args.ae_n_enc_2,
    ae_n_enc_3=args.ae_n_enc_3,
    ae_n_dec_1=args.ae_n_dec_1,
    ae_n_dec_2=args.ae_n_dec_2,
    ae_n_dec_3=args.ae_n_dec_3,
    n_input=args.n_input,
    n_z=args.n_z).to(device)

Pretrain_ae(model, dataset, y, train_loader, device)