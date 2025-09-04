# MICrONS-datacleaner

[![License](https://badgen.net/github/license/MICrONS-Milano-CoLab/MICrONS-datacleaner)](https://opensource.org/licenses/MIT)
[![PyPi_Version](https://badgen.net/pypi/v/Microns-DataCleaner)](https://pypi.org/project/Microns-DataCleaner/)

This project contains tools to work with the [MICrONS Cortical MM3 dataset](https://www.microns-explorer.org/cortical-mm3), providing a **robust interface** to interact with the nucleus data. 

## Key features 

- **Simple interface** to download and keep organized anatomical data via CAVEClient. 
- **Allows to query the synapse table in chunks** avoiding common pitfalls. 
- **Easily process nucleus annotation tables**. 
- **Automatically segment** the brain volume into cortical **layers.**
- **Tools for filtering** and constructing connectome subsets. 
- Basic interface to add functional properties, including **tuning curves and selectivity**. 

## Install 📥

```bash
pip install microns-datacleaner
```

## Using the package ⏩

- **Few lines of code** to get a full table with neurons' brain area, layer, cell_type, proofreading information, and nucleus position:

```python
#Import the lib
import microns_datacleaner as mic

#Target version and download folder
cleaner = mic.MicronsDataCleaner(datadir = "data", version=1300) 

#Download the data
cleaner.download_nucleus_data()

#Process the downloaded data and segment into layers
units, segments = cleaner.process_nucleus_data()
```

- **Filter easily!** How can we get all neurons in V1, layers L2/3 and L4 with proofread axons?

```python
units_filter = fl.filter_neurons(units, layer=['L2/3', 'L4'], proofread='ax_clean', brain_area='V1')
```

- **Robustly download synapses** between a subset of pre and post-synaptic neurons in chunks. 

```python
preids  = units_filter['pt_root_id']
postids = units_filter['pt_root_id']
cleaner.download_synapse_data(preids, postids)

#Connection problems at chunk number 23? Just restart from there
cleanerdownload_synapse_data(preids, postids, start_index=23)
```

Check the docs and our tutorial notebook just below to get started!


## Docs & Tutorials 📜 

If it is the first time working with the MICrONS data, we recommend you read our basic tutorial (also available as a Python Notebook), as well as the official documentation of the MICrONS project.  

If you want to contribute, please read our guidelines first. Feel free to open an issue if you find any problem.  

You can find a full documentation of the API and functions in the [docs](https://microns-milano-colab.github.io/MICrONS-datacleaner/). 


## Requirements 

### Dependencies 

- CaveCLIENT
- Pandas
- Numpy
- Scipy
- TQDM
- Standard transform for coordinate change (MICrONS ecosystem)

### Dev-dependencies 

- pdoc (to generate the docs)
- ruff (to keep contributions in a consistent format)



## Citation Policy 📚

If you use our code, **please consider to cite the associated repository,** as well as the [IARPA MICrONS Minnie Project](https://doi.org/10.60533/BOSS-2021-T0SY) and the [Microns Phase 3 NDA](https://github.com/cajal/microns_phase3_nda) repository. 

Our code serves as an interface for the MICrONS data. Please cite appropiate the literature for the data used following their [Citation Policy](https://www.microns-explorer.org/citation-policy). The papers may depend on the annotation tables used.

Our unit table is constructed by integrating information from the following papers:

1. [Functional connectomics spanning multiple areas of mouse visual cortex](https://doi.org/10.1038/s41586-025-08790-w). The Microns Consortium. 2025 
2. [Foundation model of neural activity predicts response to new stimulus types](https://doi.org/10.1038/s41586-025-08829-y)
3. [Perisomatic ultrastructure efficiently classifies cells in mouse cortex](http://doi.org/10.1038/s41586-024-07765-7)
4. [NEURD offers automated proofreading and feature extraction for connectomics](https://doi.org/)
5. [CAVE: Connectome Annotation Versioning Engine](https://doi.org/10.1038/s41592-024-02426-z)

## Acknowledgements

We acknowledge funding by the NextGenerationEU, in the framework of the FAIR—Future Artificial Intelligence Research project (FAIR PE00000013—CUP B43C22000800006). 


## Generating the docs

Go to the main folder of the repository, and run

```
pdoc -t docs-src/template src/microns_datacleaner/ -o docs/
```

The docs will be generated in the `docs/` folder in HTML format, which can be checked with the browser. 


