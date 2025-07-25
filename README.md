
# PHD-MS: Persistent Homology for Domains at Multiple Scales

Multiscale domain identification for spatial transcriptomic data.


## Installation

Simply install with pip:


    pip install phd_ms

Or install from source by downloading, cd to this directory, using:
     
    pip install .

## Usage/Examples

Detailed jupyter notebook tutorials are available in the examples folder. 
Or simply download the file titled 'point_and_click.py' and run the file to use our clickable graphical interface. When using the point_and_click.py interface, update the directory where your data is stored by changing this line:
```python
DATA = '/home/pbeamer/Documents/graphst/visium_hne_graphst'
```

Here we'll show a simple example with Visium DLPFC data, using default parameters.
First, import necessary components.
```python
import phd-ms
import scanpy
```
Preprocessing steps here (note that we assume a spatially-aware embedding has already been computed for your data):
```python
INPUT_FILE= '/home/pbeamer/Documents/graphst/adata_151673
phd_ms.tl.preprocess_leiden(INPUT_FILE,output_file=INPUT_FILE)
```
Compute persistent homology and plot 10 most prominent multiscale domains:
```python
cluster_complex,clusterings= phd_ms.tl.cluster_filtration(adata)
phd_ms.tl.map_multiscale(adata.obsm['spatial'],cluster_complex,clusterings,num_domains=10)
```
