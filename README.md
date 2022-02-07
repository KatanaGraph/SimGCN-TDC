# SimGCN-TDC
This repo contains gnn pipeline for submission to TDC benchmark challenge.

# Installation 

Use conda to install all the required dependencies. 

```
conda env create -f env.yml
conda activate tdc-task
pip install git+https://github.com/bp-kelley/descriptastorus 

```

Note that you need to ensure pyg version is 2.0.2. The results might not be reproducible in other versions.

# Reproducibility Instructions

To reproduce the results for a specific dataset, use the following script.

```
python run.py --dataset hERG --verbose False
```