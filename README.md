This is a Pytorch implementation of ASD-VAE:  Incomplete Graph Learning via Attribute-Structure Decoupled Variational Auto-Encoder. (WSDM 2024)

# Incomplete Graph Learning via Attribute-Structure Decoupled Variational Auto-Encoder

More details of the paper and dataset will be released after it is published.


# The Code

## Requirements

Following is the suggested way to install the dependencies:

    conda install --file ASD-VAE.yml

Note that ``pytorch >=1.10``.

## Folder Structure

```tex
└── code-and-data
    ├── data                    # Including datasets-Amazon and Planetoid
    ├── models                  # The core source code of our model SPGCL
    │   |──  _init_.py          # Initialization file for models
    │   |──  ASD_VAE.py         # Including fix model in ASD-VAE    
    │   |──  gnn.py             # Including predict models of GNNs 
    ├── structure_data          # Contains Euclidean structute by one-hot or d2c 
    ├── utils                   # Defination of auxiliary functions for running
    │   |──  _init_.py          # Initialization file for utils
    │   |──  data_utils.py      # Data load and process preparation    
    │   |──  missing.py         # Generate two types of missing masks
    │   |──  plot_TSNE.py       # Visualization of graph embedding
    │   |──  gen_structure.py   # Generate structure by using d2c code 
    │   |──  args.py            # Settings about models and loading configure files
    ├── CONFIG                  # Datasets hyperparameter settings
    ├── mian.py                 # This is the main file
    ├── train.py                # Trainer for ASD-VAE 
    ├── ASD-VAE.yml             # The python environment needed for ASD-VAE
    └── README.md               # This document
```

## Datasets

Download Cora & Citeseer datasets from https://github.com/kimiyoung/planetoid; 

Download AmaComp & AmaPhoto datasets from https://github.com/shchur/gnn-benchmark.

## Configuration

Step 1:  Parameter settings for different datasets are all in  `./CONFIG/` 

Step 2:  Important parameters in the configuration are as follows (take Cora as example):

```tex
nhid = 16               # The hidden unit of Graph Encoder and Predictor
beta_2 = 0.2            # The decoupling coefficient of negative edges, i.e., \beta_2
filter = Katz           # The filter used in Predictor and Graph Encoder, i.e., Res
gan_alpha_missing = 10  # Penalty coefficient for missing attribute, i.e., \alpha_D
```


##  Train and Test

Replace `"your_own_data_name"` with your own dataset name (e.g., Cora) and you can start training and testing your model.

For train your own model, you can use main.py with some options to specify dataset, missing type, missing rate, and hyper-parameters:

- ```tex
  ！python main.py --dataset='cora'     --type='uniform' --rate=0.1  
  ！python main.py --dataset='citeseer' --type='struct'  --rate=0.9
  ```

  All the parameter settings are in `utils.args.py` and `CONFIG/` 

We provide more options for you for further study:

- ```tex
  --attack_ratio        # The ratio for attacking graph adjacency matrix
  --r_type              # Attack type such as flip/add/remove
  --use_lcc             # Test on the connectivity graph
  --split_train         # Use one optimizer for fix and predict
  --profile             # Experiment setting: node classification or profile
  ```
 
 ## Main Baseline Codes
  - GCNMF:  "Graph Convolutional Networks for Graphs Containing Missing Features" (https://github.com/marblet/GCNmf)
  - SAT: "Learning on Attribute-Missing Graphs" (https://github.com/xuChenSJTU/SAT-master-online)
  - SVGA: "Accurate Node Feature Estimation with Structured Variational Graph Autoencoder" (https://github.com/snudatalab/SVGA)
  - GNN-AC: "Heterogeneous Graph Neural Network via Attribute Completion" (https://github.com/liangchundong/HGNN-AC)
  - ITR: "Initializing Then Refning: A Simple Graph Attribute Imputation Network" (https://github.com/WxTu/ITR)
  - SGC: "Simplifying Graph Convolutional Networks" (https://github.com/Tiiiger/SGC)
  - AGE: "Adaptive Graph Encoder for Attributed Graph Embedding" (https://github.com/thunlp/AGE)
