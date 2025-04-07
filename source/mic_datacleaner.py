import os
from pathlib import Path

from caveclient import CAVEclient

import downloader as dwn
import pandas as pd


class MicronsDataCleaner:

    homedir = Path().resolve() 
    datadir = "data" 

    tables_2_download = ["nucleus_detection_v0", "baylor_log_reg_cell_type_coarse_v1", "baylor_gnn_cell_type_fine_model_v2", "aibs_metamodel_celltypes_v661"] 


    def __init__(self):
        self.data_storage = f"{self.homedir}/{self.datadir}"


    def initialize_client(self, version = None):
        self.client = CAVEclient('minnie65_public') 
        if version != None:
            self.client.version = version

    def download_nucleus_data(self):
        dwn.download_nucleus_data(self.client,f"{self.data_storage}/raw",  self.tables_2_download)

    def download_synapse_data(self, presynaptic_set, postsynaptic_set, neurs_per_steps = 500, start_index=0, max_retries=10, delay=5, drop_synapses_duplicates=False):
        dwn.connectome_constructor(self.client, presynaptic_set, postsynaptic_set, f"{self.data_storage}/raw/synapses",
                                   neurs_per_steps = neurs_per_steps, start_index=start_index, max_retries=max_retries, delay=delay, drop_synapses_duplicates=drop_synapses_duplicates)
        return

    def merge_synapses(self):
        dwn.merge_connection_tables(f"{self.data_storage}/raw")
        return

    def process_nucleus_data(self):
        nucleus  = pd.read_csv(f"{self.data_storage}/raw/nucleus_detection_v0.csv")
        celltype = pd.read_csv(f"{self.data_storage}/raw/aibs_metamodel_celltypes_v661.csv")
        nucleus_merged = dwn.merge_nucleus_with_cell_types(nucleus, celltype)
        nucleus_merged = dwn.transform_positions(nucleus_merged)

        segments = dwn.divide_volume_into_segments(nucleus_merged)
        segments = dwn.merge_segments_by_layer(segments)

        return nucleus_merged, segments


    
