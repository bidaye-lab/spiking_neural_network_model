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

## Custom neuron names
Custom neuron names used in the lab can be edited in this 
[spread sheet in the OneDrive](https://maxplanckflorida-my.sharepoint.com/:x:/g/personal/murakamik_mpfi_org/EeX_NEJ2kaVMvcHdbHPZkPcBG9IwOMWwkEingWCFmnv_SA?e=azcslm).
The shared document is intended as the reference for the entire lab.
The file `flywire_ids_630.xlsx` in this git repository may be outdated.

There are two types of spread sheets:
- pairs: If the sister neurons are known, flywire IDs for each of them is entered in a separate column. The suffixes `_r` and `_l` will be automatically appended to the neuron names.
- single: If the sister neuron is not known, 
there is one flywire ID column. The name is used as is.

Please make sure to follow the consenus in the lab about naming conventions when assigning names for neurons here.

## More calculations
Many simultaneous and delayed stimalation experiments involving
walk, stop, and sensory neurons have already been peformed.
See notebook `\\mpfi.org\public\sb-lab\nico\flywire_brian2\coac_dly_experiments.ipynb`
and the corresponding `results\coac` and `results\dly` folders.