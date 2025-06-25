import os
from pathlib import Path

from caveclient import CAVEclient

import downloader as down
import processing as proc
import pandas as pd


class MicronsDataCleaner:
    """
    The main class to download and clean data from the Microns Dataset. 
    """

    """
    Path where the code is executed
    """
    homedir = Path().resolve() 
    """
    Subfolder aimed to contain the downloaded data
    """
    datadir = "data" 

    """
    List of tables to download from the set
    """
    tables_2_download = ["nucleus_detection_v0", "baylor_log_reg_cell_type_coarse_v1", "baylor_gnn_cell_type_fine_model_v2", "aibs_metamodel_celltypes_v661", 
                         "coregistration_manual_v4", "functional_properties_v3_bcm", "nucleus_functional_area_assignment",
                         "proofreading_status_and_strategy"] 


    def __init__(self, datadir="data"):
        """
        Initialize the class and makes sure subfolders to download exist.

        Parameters:
        -----------
            datadir: string, optional. Default to 'data'. Points to the folder where information will be downloaded.
        """

        self.datadir = datadir
        self.data_storage = f"{self.homedir}/{self.datadir}"
        # Ensure directories exist
        os.makedirs(self.data_storage, exist_ok=True)
        os.makedirs(f"{self.data_storage}/raw", exist_ok=True)
        os.makedirs(f"{self.data_storage}/raw/synapses", exist_ok=True)


    def initialize_client(self, version = None):
        """
        Initialize a CAVEClient with the desired version.

        Parameters:
        -----------
            version: optional, allows to fix the version to download. If left at None, it points to the last one.

        """

        self.client = CAVEclient('minnie65_public') 
        if version is not None:
            self.client.version = version

    def download_nucleus_data(self):
        """
        Downloads all the tables indicated in the cleaner.tables to download 
        """

        down.download_nucleus_data(self.client,f"{self.data_storage}/raw",  self.tables_2_download)

    def download_synapse_data(self, presynaptic_set, postsynaptic_set, neurs_per_steps = 500, start_index=0, max_retries=10, delay=5, drop_synapses_duplicates=False):
        """
        Downloads all the synapses for the specifiied pre- and post- synaptic steps.

        Parameters:
        -----------
            client: CAVEclient needed to access MICrONS connectomics data
            presynaptic_set: 1-d array of non repeated root_ids of presynaptic neurons for which to extract postsynaptoc connections in postynaptic_set
            postsynaptic_set: 1-d array of non repeated root_ids of postsynaptic neurons for which to extract presynaptic connections in presynaptic_set
            neurs_per_steps: optional, defaults to 500. Number of postsynaptic neurons for which to recover presynaptic connectivity per single call to the connectomics
                database. Since the connectomics database has a limit on the number of connections you can query at once
                this iterative method optimises querying multiple neurons at once, as opposed to each single neuron individually,
                while also preventing the queries from crashing. I have tested that for a presynaptic set of around 8000 neurons
                you can reliably extract the connectivity for around 500 postsynaptic neurons at a time.
            start_index: optional, defaults to 0. If previous download was interrupted, one can manually set the index of the last file downloaded to continue
                from that point on. For any fresh download it should be kept 0.
            max_retries: optional, defaults to 10. The number of times to retry if the server is not responding before giving up.
            drop_synapses_duplicates: optional, defaults to True. If true, it merges all the synapses between neuron i-th and j-th to a single connection in which
                the synapse_size is the total sum of all synapse sizes between those two elements.
        """
        down.connectome_constructor(self.client, presynaptic_set, postsynaptic_set, f"{self.data_storage}/raw/synapses",
                                   neurs_per_steps = neurs_per_steps, start_index=start_index, max_retries=max_retries, delay=delay, drop_synapses_duplicates=drop_synapses_duplicates)
        return

    def merge_synapses(self):
        """
        Merges all the batches of the downloaded synapses. 
        """
        down.merge_connection_tables(f"{self.data_storage}/raw")
        return

    def process_nucleus_data(self):
        """
        Processes all the downloaded nucleus data to generate a unified units_table. This includes information on 
        brain area, functional data, proofreading, as well as a layer segmentation.

        Parameters:
        -----------
            None
        
        Returns:
        --------
            nucleus_merged: a unit_table with all the nucleus information processed.
            segments: an array with the positions of all segments
        """
        try:

            #Read all the downloaded data
            nucleus   = pd.read_csv(f"{self.data_storage}/raw/nucleus_detection_v0.csv")
            celltype  = pd.read_csv(f"{self.data_storage}/raw/aibs_metamodel_celltypes_v661.csv")
            areas     = pd.read_csv(f"{self.data_storage}/raw/nucleus_functional_area_assignment.csv")
            funcprops = pd.read_csv(f"{self.data_storage}/raw/functional_properties_v3_bcm.csv")
            proofread = pd.read_csv(f"{self.data_storage}/raw/proofreading_status_and_strategy.csv")

            #Call all the merge functions
            nucleus_merged = proc.merge_nucleus_with_cell_types(nucleus, celltype)
            nucleus_merged = proc.merge_brain_area(nucleus_merged, areas)
            nucleus_merged = proc.merge_proofreading_status(nucleus_merged, proofread)
            nucleus_merged = proc.merge_functional_properties(nucleus_merged, funcprops)

            #Get the correct positions
            nucleus_merged = proc.transform_positions(nucleus_merged)

            #Segment the data and add the information about layers
            segments = proc.divide_volume_into_segments(nucleus_merged)
            segments = proc.merge_segments_by_layer(segments)

            proc.add_layer_info(nucleus_merged, segments)

            #Clean the resulting table and return
            nucleus_merged = nucleus_merged[nucleus_merged['pt_root_id'] > 0]
            nucleus_merged = nucleus_merged.drop_duplicates(subset='pt_root_id')

            return nucleus_merged, segments
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Required data file not found: {e}")
            raise
        except Exception as e:
            raise RuntimeError(f"Error processing nucleus data: {e}")