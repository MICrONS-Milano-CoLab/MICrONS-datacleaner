# MICrONS-datacleaner

# Overview
This project provides tools for working with the MICrONS Minnie65 dataset, allowing users to:
- Download neuroanatomical data from the MICrONS database
- Process and merge neuronal nucleus data
- Build connectome subsets for specific neuron groups
- Segment the brain volume into cortical layers

# Structure
downloader.py: Functions for downloading data from the MICrONS database
processing.py: Functions for processing, merging, and analyzing neural data
mic_datacleaner.py: Wrapper class that simplifies the use of download and processing functions

# Requirements 
- CaveCLIENT
- Pandas
- Numpy
- Scipy
- H5Py
- TQDM
- Standard transform for coordinate change (MICrONS ecosystem)
