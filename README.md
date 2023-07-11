# Leaky-integrate-and-fire model based on MANC v1.0 connectome

## Installation

### Jupyter notebooks

The workflow is designed for Jupyter notebooks.
In the following, it is assumed that [anaconda](https://www.anaconda.com/download/) is installed on the computer.
Jupyter notebooks will be installed together with the other python packages below.

Alternatively, notebooks can be handled with [JupyterLab](https://jupyter.org/) or [Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
These programs make working with python code significantly more convenient.

## Download code

To download, click on _Code_ at the top of this web page, download the repository as `.zip` and extract

### Dependencies

Install the the following through the _Individual components_ tab in the [Visual Studio Installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/):

- `MSVC v143 VS2022 C++ x64/x86 built tools` (or latest version)
- `Windows 10 SDK` (latest version)

The required python packages are collected in `environment.yml`, which can be installed via `conda`.
Create a new environment by opening an Anaconda terminal, change the working directory to this folder and and call

`conda env create -f environment.yml -n brian2`

### Run

In the Anaconda terminal, run `conda activate brian2`.
Change the working directory to the downloaded folder (`cd` command) and run `jupyterlab` (recommended)
or `jupyter notebook`.
Test the installation by opening and running `example.ipynb`.
Watch the Anaconda terminal for possible warnings related to `brian2`.

### Generatsing Neuron Data

There is an additional notebook attached to this repository called `MANC_data_crawler.ipynb`. Run the function inside after installing needed dependencies. You will need to provide your own authentication token from the Neuprint website. The script will generate two files `output.csv` and `output.parquet`. 

### MANC vs. Flywire Data

In the `Computational Setup` section of the `example.ipynb`, you will see two cells. The first cell (which is uncommented) contains the environment and path setup to use the MANC V1.0 Data set in this Brian2 model. If you'd wish to use Flywire data, comment out the first cell in `Computational Setup` and uncomment the second cell to use the Flywire environment file path setup.

## Using multiple CPUs

The computational load can be distributed over multiple CPU threads.
To choose the number of threads, set `n_procs` to a number not higher than available on your computer.
More threads will speed up the computation, but also increase memory usage.

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

## Graphs

The `example.ipynb` shows how to summarize a computational experiment in a single graph.
This is done in two steps:

1. The full connectome is loaded in a `DiGraph` object using [NetworkX](https://networkx.org/).
2. The subset of neurons active in a given experiment is selected from the full graph and written to disk.
   The resulting `*.gexf` file can be loaded for visualization with [Gephi](https://gephi.org/).
