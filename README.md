# Leaky-integrate-and-fire model based on Flywire connectome

## Installation
### Jupyter notebooks
The workflow is designed for Jupyter notebooks.
The easiest way to install Jupyter notebooks is via [anaconda](https://www.anaconda.com/download/).

Alternatively, notebooks can be handled with [JupyterLab](https://jupyter.org/) or [Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
These programs make working with python code significantly more convenient.

### Dependencies
The required python packages are collected in `environment.yml`, which can be installed via `conda`.
Create a new environment by opening an Anaconda terminal, change the working directory to this folder and and call

`conda env create -f environment.yml -n brian2`

### Test installation
Test the installation by running `example.ipynb`.
