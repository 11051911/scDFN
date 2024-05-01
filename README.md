# scDFN: Enhancing Single-cell RNA-seq Clustering with Deep Fusion Networks

## Introduction

Single-cell RNA sequencing (scRNA-seq) technology can be used to perform high-resolution analysis of the transcriptomes of individual cells. Therefore, its application has gained popularity for accurately analyzing the ever-increasing content of heterogeneous single-cell datasets. Central to interpreting scRNA-seq data is the clustering of cells to decipher transcriptomic diversity and infer cell behavior patterns. Although clustering plays a key role in the subsequent analysis of single-cell transcriptomics, its complexity necessitates the application of advanced methodologies capable of resolving the inherent heterogeneity and limited gene expression characteristics of single-cell data. In this study, we introduced a novel deep learning-based algorithm for single-cell clustering, designated scFCN, which can significantly enhance the clustering of scRNA-seq data through a fusion network strategy. The scFCN algorithm applies a dual mechanism involving an autoencoder to extract attribute information and an improved graph autoencoder to capture topological nuances, integrated via a cross-network information fusion mechanism complemented by a triple self-supervision strategy. This fusion is optimized through a holistic consideration of four distinct loss functions. A comparative analysis with five leading scRNA-seq clustering methodologies across multiple datasets revealed the superiority of scFCN, as determined by better Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI) metrics. Additionally, scFCN demonstrated robust multi-cluster dataset performance and exceptional resilience to batch effects. Ablation studies highlighted the key roles of the autoencoder and the improved graph autoencoder components, along with the critical contribution of the four joint loss functions to the overall efficacy of the algorithm. Through these advancements, scFCN set a new benchmark in single-cell clustering and can be used as an effective tool for the nuanced analysis of single-cell transcriptomics. The source code of scFCN is publicly available at https://github.com/11051911/scDFN.

## Environment

* Anaconda
* python 3.8+
## Dependency

*Pytorch (2.0+)
*Numpy  1.26.4
*Torchvision 0.17.2
*Matplotlib 3.8.4
*h5py 3.11.0
*Matplotlib 3.8.4
*pandas 2.2.2
*numpy 1.26.4
*scanpy 1.10.1
*scipy 1.12.0

## Installation

1. Download and install Anaconda.
   ```git clone https://github.com/11051911/scDFN.git```

2. Create the prosperousplus environment.

   ```conda create -n scDFN python=3.9.13```

3. Activate the prosperousplus environment and install the dependencies.

   ```conda activate prosperousplus```

   ```pip install or conda install```

## Usage

Here we provide an implementation of Enhancing single-cell RNA-seq Clustering with Deep Fusion Networks (scDFN) in PyTorch, along with an execution example on the goolam dataset. The repository is organised as follows:

- `scDFN.py`: defines the architecture of the whole network.
- `IGAE.py`: defines the improved graph autoencoder.
- `AE.py`: defines the autoencoder.
- `opt.py`: defines some hyper-parameters.
- `train.py`: the entry point for training and testing.

Finally, `main.py` puts all of the above together and may be used to execute a full training run on goolam.

## Examples: 

### Clustering:
We got the pre-training files in AE, GAE and pretrain respectively, and then trained in the train folder. We provided the pre-training files of goolam.

```python main.py```


## Output:
NMI and ARI values ​​for clustering.

