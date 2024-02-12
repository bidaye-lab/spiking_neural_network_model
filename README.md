# Leaky-integrate-and-fire model based on connectome data

This repository contains code to simulate a spiking neural network model based on the connectome data of the fruit fly.
It is based on the [original model](https://github.com/philshiu/Drosophila_brain_model)
but is further developed by the Bidaye lab.

For more information on the general structure of this repo, 
see this [template repo](https://github.com/bidaye-lab/template_data_pipelines).

# Overview
Details about how to use the model are given in the separate workflow scripts:
|file|content|
|---|---|
|[example.py](scripts/example.py)|General usage|
|[graph_for_cytoscape.ipynb](notebooks/old/graph_for_cytoscape.ipynb)| Visualizations for cytoscape|
|[heatmap_2freq.ipynb](notebooks/old/heatmap_2freq.ipynb)| Custom 2D frequency comparison|


# Installation

```
# get source code
git clone https://github.com/nspiller/spiking_neural_network_model 
cd spiking_neural_network_model 

# create conda environment with necessary dependencies
conda env create -n spiking_neural_network_model -f environment.yml
conda activate spiking_neural_network_model

# install project code as local local python module
pip install -e .

# convert scripts to notebooks
jupytext --sync scripts/*.py
```
## Windows only
To significantly speed up the simulations, install the the following through the _Individual components_ tab in the [Visual Studio Installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/):
- `MSVC v143 VS2022 C++ x64/x86 built tools` (or latest version)
- `Windows 10 SDK` (latest version)

See
[official Brian2 documentation](https://brian2.readthedocs.io/en/stable/introduction/install.html#requirements-for-c-code-generation) on "Requirements for C++ code generation" for more details.
