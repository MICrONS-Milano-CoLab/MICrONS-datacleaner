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

    def process_nucleus_data(self):
        nucleus  = pd.read_csv(f"{self.data_storage}/raw/nucleus_detection_v0.csv")
        celltype = pd.read_csv(f"{self.data_storage}/raw/aibs_metamodel_celltypes_v661.csv")
        nucleus_merged = dwn.merge_nucleus_with_cell_types(nucleus, celltype)

        return dwn.transform_positions(nucleus_merged)


    
