# scLinformer

**scLinformer: an efficient and transferable framework for representation learning of cells and genes for single-cell analysis**

![image-20260409203123935](static/img/fig.png)

## Installation

**scLinformer**

Only `scLinformer` needs to be installed if you only use the model.

```shell
git clone https://github.com/TRCDHH/scLinformer.git
conda create -n scLinformer python==3.10
conda activate scLinformer
pip install .
```

**Agent (Optional)**

The following components are required **only if you want to run the agent service**.

[TRCDHH/RNAgent](https://github.com/TRCDHH/RNAgent)

## Quick Start

The model training, evaluation, and metric computation are implemented in train.py.

```
python train.py
```

## Data Available

- Immune https://figshare.com/ndownloader/files/25717328
- Lung atlas https://figshare.com/ndownloader/files/24539942
- BMMC https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122
