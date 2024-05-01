import torch
import random
import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy import sparse as sp
from sklearn.cluster import SpectralClustering


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('Epoch_{}'.format(epoch),  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          )
    return nmi, ari


def parameter(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("sum:" + str(k))
    return str(k)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]),
        torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset


def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):
    lr = max(init_lr * (0.9 ** (iteration // adjust_epoch)), max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def read_dataset(File1=None, File2=None, File3=None, File4=None, format_rna="table", format_epi="table", transpose=True,
                 state=0):
    adata = adata1 = None

    if File1 is not None:
        if format_rna == "table":
            adata = sc.read(File1)
        else:  # 10X format
            adata = sc.read_mtx(File1)

        if transpose:
            adata = adata.transpose()

    if File2 is not None:
        if format_rna == "table":
            adata1 = sc.read(File2)
        else:  # 10X format
            adata1 = sc.read_mtx(File2)

        if transpose:
            adata1 = adata1.transpose()

    label_ground_truth = []
    label_ground_truth1 = []
    if state == 0:
        if File3 is not None:
            Data2 = pd.read_csv(File3, header=0, index_col=0)
            label_ground_truth = Data2['Group'].values

        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_csv(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Group'].values

        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    elif state == 1:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['cell_line'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['cell_line'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    elif state == 3:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['Group'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Group'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    else:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['Cluster'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Cluster'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    adata.obs['Group'] = label_ground_truth
    adata.obs['Group'] = adata.obs['Group'].astype('category')

    adata1.obs['Group'] = label_ground_truth
    adata1.obs['Group'] = adata1.obs['Group'].astype('category')

    print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata, adata1, label_ground_truth, label_ground_truth1


def normalized(adata, filter_min_counts=True, size_factors=True, highly_genes=None,
               normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    """ if size_factors:
        #adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        adata.obs['size_factors'] = np.log( np.sum( adata.X, axis = 1 ) )
    else:
        adata.obs['size_factors'] = 1.0 """

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


def get_adj(count, k=10, pca=50, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)

    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)


def clustering(args, z, y, adjn1):
    labels_k = KMeans(n_clusters=args.n_clusters, n_init=20).fit_predict(z.data.cpu().numpy())
    labels_s = SpectralClustering(n_clusters=args.n_clusters, affinity="precomputed", assign_labels="discretize",
                                  n_init=20).fit_predict(adjn1.data.cpu().numpy())
    labels = labels_s if (np.round(metrics.normalized_mutual_info_score(y, labels_s), 5) >= np.round(
        metrics.normalized_mutual_info_score(y, labels_k), 5)
                          and np.round(metrics.adjusted_rand_score(y, labels_s), 5) >= np.round(
                metrics.adjusted_rand_score(y, labels_k), 5)) else labels_k
    nmi, ari = eva(y, labels)
    centers = computeCentroids(z.data.cpu().numpy(), labels)
    return nmi, ari, centers


# def eva(y_true, y_pred):
#     nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
#     ari = ari_score(y_true, y_pred)
#     return nmi, ari


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))  # torch.unique
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])


#SF
import h5py
import utils as utils

def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data

def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = utils.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = utils.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                                exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def prepro(filename):
    data_path = filename
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label