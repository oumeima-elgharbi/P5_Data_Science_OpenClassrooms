# Project 5 Data Science OpenClassrooms

## Unsupervised Learning : Clustering

Author : Oumeima EL GHARBI.

Date : September, October 2022.

### Context

We have dataset provided by Olist, the largest department store in Brazilian marketplaces, and we want to make a **Customer Segmentation**.
This is a project about **Unsupervised Learning** in which we will use **Clustering Algorithms**.

Using a RFM Segmentation, we have tried these clustering algorithms :

- Centroid-based Clustering : **K-Means**
- Hierarchical Clustering : **Agglomerative Clustering**
- Density-based Clustering : **DBSCAN**

We also computed a RFM (Recency, Frequency, Monetary) Score and tried a Segmentation based on Personae.

### Dataset folder

- Create a folder **dataset**

- Create a folder **dataset/source**
- Create a folder **dataset/cleaned**
- Create a folder **dataset/simulation**

- Create a folder **dataset/simulation/experiment_1**
- Create a folder **dataset/simulation/experiment_2**

- Download the zip folder at this address and unzip it in the **source** folder :
  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/download?datasetVersionNumber=2

Source  : https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

#### Attention from the data source

An order might have multiple items.
Each item might be fulfilled by a distinct seller.
All text identifying stores and partners where replaced by the names of Game of Thrones great houses.


### Model folders

- Create a folder **model**

### Libraries

Install the python libraries with the same version :

```bash
pip install -r requirements.txt
```

### Execution

```bash
run P5_01_exploration.ipynb
run P5_02_essais.ipynb
run P5_03_simulation.ipynb
```
