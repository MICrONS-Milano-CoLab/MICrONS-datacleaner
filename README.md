# MICrONS-datacleaner

# Overview
This project provides tools for working with the MICrONS Minnie65 dataset, allowing users to:
- Download neuroanatomical data from the MICrONS database
- Process and merge neuronal nucleus data
- Build connectome subsets for specific neuron groups
- Segment the brain volume into cortical layers

# Structure
- `mic_datacleaner`.py: Wrapper class that simplifies the use of download and processing functions.
- `downloader.py`: Functions for downloading data from the MICrONS database
- `processing.py`: Functions for processing, merging, and analyzing neural data

# Requirements 

## Required 

- CaveCLIENT
- Pandas
- Numpy
- Scipy
- TQDM
- Standard transform for coordinate change (MICrONS ecosystem)

## Optional

- pdoc (to generate the docs)
- ruff (to keep contributions in a consistent format)

# Fast Docs

## How to use 

(To be written)
See the `tutorial` notebook.

## Generating the docs

Go to the main folder of the repository, and run

```
pdoc -t docs/template source/mic_datacleaner.py -o docs/html
```

The docs will be generated in the `docs/html` folder in HTML format, which can be checked with the browser. If you need the docs for all the files, and not only the class, use `source/*.py` instead of `source/mic_datacleaner.py` above.