# SimGCN-TDC
This repo contains gnn pipeline for submission to TDC benchmark challenge.

# Installations 

Use conda to install all the required dependencies. To install conda, follows these steps.

```
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Use the `env.yml` file to populate the environment as follows. 

```
conda env create -f env.yml
conda activate tdc-task
pip install git+https://github.com/bp-kelley/descriptastorus 

```

Note that you need to ensure pyg version is 2.0.2. The results might not be reproducible in other versions.

# Results

The results are summarized in the Table below. Also refer to the report for more details. 

![Alt text](results.png?raw=true "Title")

# Reproducibility Instructions

We conducted all our experiments in AWS `c5.12xlarge` instances (CPU only). To reproduce the results for a specific dataset, use the following script.

```
python run.py --dataset hERG --verbose False
```