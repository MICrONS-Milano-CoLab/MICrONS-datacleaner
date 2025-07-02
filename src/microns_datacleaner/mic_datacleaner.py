import os
from pathlib import Path
import pandas as pd

from caveclient import CAVEclient

from . import downloader as down
from . import processing as proc

import requests

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
    Version of the client we are using
    """
    version = 1300

    """
    List of tables to download from the set
    """
    #tables_2_download = ["nucleus_detection_v0", "baylor_log_reg_cell_type_coarse_v1", "baylor_gnn_cell_type_fine_model_v2", "aibs_metamodel_celltypes_v661", 
    #                     "coregistration_manual_v4", "functional_properties_v3_bcm", "nucleus_functional_area_assignment",
    #                     "proofreading_status_and_strategy"] 


    def __init__(self, datadir="data", version=1300, custom_tables={}, download_policy='minimum', extra_tables=[]):
        """
        Initialize the class and makes sure subfolders to download exist. Configures the tables to be downloaded (except synapses) via
        a download policy.

        Parameters:
        -----------
            datadir: string, optional. 
                Defaults to 'data'. Points to the folder where information will be downloaded.
            custom_tables: dict, optional. 
                Used to override the default tables used to construct the unit table in a given version. The keys for the tables
                to be overrided are 'celltype' for the nucleus classification scheme, 'proofreading' for the prooreading table,
                'brain_areas' for assigned brain areas, 'func_props' for functional properties, and 'coreg' for the coregistration table.
            download_policy: str, optional
                Used to set how the tables should be downloaded. 'minimum' (the default) only downloads the minimum amount of tables necessary 
                to construct our unit table. 'all' gets all of them. 'extra' gets the same as 'minimum' plus the tables specified in `extra_tables`.
            extra_tables: list, optional
                List of extra table names to be downloaded. See the download_police for more information.
        """

        #Places we the data is going to be downloaded
        self.version = version
        self.datadir = datadir
        self.data_storage = f"{self.homedir}/{self.datadir}/{self.version}"

        # Ensure directories exist
        os.makedirs(self.data_storage, exist_ok=True)
        os.makedirs(f"{self.data_storage}/raw", exist_ok=True)
        os.makedirs(f"{self.data_storage}/raw/synapses", exist_ok=True)

        #Initialize the CAVEClient with the user-specified version 
        self._initialize_client(version)

        #Set the tables to download according to the version
        self._configure_download_tables(version, custom_tables, download_policy, extra_tables)

    def _configure_download_tables(self, version, custom_tables, download_policy, extra_tables):
        """
        Internal function that configures the tables to be downloaded according to the selected version and download policy.
        See the constructor for information on custom_tables and possible policies. This function is not intended for the user. 
        """

        self.tables = {}

        match version:
            case 661:
                self.tables['nucleus']      = "nucleus_detection_v0"
                self.tables['celltype']     = "baylor_gnn_cell_type_fine_model_v2"
                self.tables['proofreading'] = "proofreading_status_public_release"
                self.tables['brain_areas']  = None 
                self.tables['func_props']   = None 
                self.tables['coreg']        = "coregistration_manual_v3"

            case 1300:
                self.tables['nucleus']      = "nucleus_detection_v0"
                self.tables['celltype']     = "aibs_metamodel_celltypes_v661"
                self.tables['proofreading'] = "proofreading_status_and_strategy"
                self.tables['brain_areas']  = "nucleus_functional_area_assignment"
                self.tables['func_props']   = "functional_properties_v3_bcm"
                self.tables['coreg']        = "coregistration_manual_v4"

        #Override all defaults with user-provided config_table
        for key in custom_tables:
            self.tables[key] = custom_tables[key]

        #Set the tables that we will need to download.
        match download_policy:
            case 'minimum':
                #Default gets only the ones we will use to generate our unit table 
                self.tables_2_download = list(self.tables.values()) 
            case 'all':
                #All gets all of them for this version
                self.tables_2_download = self.get_table_list()
            case 'extra':
                #Get the minimum ones + the tables specified in the extra array
                self.tables_2_download = list(self.tables.values()) + extra_tables 
            case 'custom':
                #Get whatever the user says, at their own risk
                self.tables_2_download = extra_tables 
            #Any other string is an error
            case _: 
                raise ValueError("`download_tables` must be either `default`, `all`, `extra`, or `custom`")

        #Eliminate any 'None' value that could have appeared
        self.tables_2_download = [x for x in self.tables_2_download if x is not None] 


    def _initialize_client(self, version):
        """
        Initialize a CAVEClient with the desired version.

        Parameters:
        -----------
            version: optional, allows to fix the version to download. If left at None, it points to the last one.

        """

        try:
            self.client = CAVEclient('minnie65_public') 
            self.client.version = version
        except requests.HTTPError as excep:
            if '503' in str(excep):
                print("HTTP error 503: the MICrONS server is temporarily unavailable.")
                print("Client cannot be used. New data cannot be downloaded.")
                print("Try again later.")
                print("Data processing can still be performed.")
        
        return

    def get_table_list(self):
        """
        Returns a complete list of the CAVEClient available tables for the selected version
        """
        return self.client.materialize.get_tables()

    def download_nucleus_data(self):
        """
        Downloads all the tables indicated in the cleaner.tables to download 
        """

        down.download_nucleus_data(self.client,f"{self.data_storage}/raw/",  self.tables_2_download)

        return

    def download_tables(self, custom_tables):
        """
        Downloads the specified tables

        Parameters
        ----------
            custom_tables: list, the names of the tables to be downloaded
        """

        down.download_nucleus_data(self.client,f"{self.data_storage}/raw/",  custom_tables) 

        return


    def download_synapse_data(self, presynaptic_set, postsynaptic_set, neurs_per_steps = 500, start_index=0, max_retries=10, delay=5, drop_synapses_duplicates=True):
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

    def merge_synapses(self, syn_table_name):
        """
        Merges all the batches of the downloaded synapses. 
        """
        down.merge_connection_tables(f"{self.data_storage}/raw", syn_table_name)
        return

    def process_nucleus_data(self, with_functional=True):
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
            nucleus   = pd.read_csv(f"{self.data_storage}/raw/{self.tables['nucleus']}.csv")
            celltype  = pd.read_csv(f"{self.data_storage}/raw/{self.tables['celltype']}.csv")
            proofread = pd.read_csv(f"{self.data_storage}/raw/{self.tables['proofreading']}.csv")

            if self.tables['brain_areas'] is not None: 
                areas     = pd.read_csv(f"{self.data_storage}/raw/{self.tables['brain_areas']}.csv")
            if self.tables['func_props'] is not None and with_functional: 
                funcprops = pd.read_csv(f"{self.data_storage}/raw/{self.tables['func_props']}.csv")

            #Call all the merge functions. First, cell types
            nucleus_merged = proc.merge_nucleus_with_cell_types(nucleus, celltype)

            #Then, brain area. In some early versions of the dataset this is not provided
            if self.tables['brain_areas'] is not None: 
                nucleus_merged = proc.merge_brain_area(nucleus_merged, areas)
            else:
                #TODO predict brain area using a classifier
                nucleus_merged['brain_area'] = 'not_available'

            #Proofreading info
            nucleus_merged = proc.merge_proofreading_status(nucleus_merged, proofread, self.version)

            #Finally, functional properties. In some versions, these need to be added from the functional data 
            if self.tables['func_props'] is not None and with_functional: 
                nucleus_merged = proc.merge_functional_properties(nucleus_merged, funcprops)
            #else:
                #TODO estimate from functional data
                #nucleus_merged['pref_ori'] = 0 
                #nucleus_merged['gOSI'] = 0. 


            #Get the correct positions
            nucleus_merged = proc.transform_positions(nucleus_merged)

            #Segment the data and add the information about layers
            segments = proc.divide_volume_into_segments(nucleus_merged)
            segments = proc.merge_segments_by_layer(segments)

            proc.add_layer_info(nucleus_merged, segments)

            #Clean the resulting table by eliminating multisoma objects
            nucleus_merged = nucleus_merged[nucleus_merged['pt_root_id'] > 0]

            #Count the number of each pt_root_id. Then use the id column as a placeholder to add 
            #that count to each row of the table. Since root_id_counts is like another column
            #of the table, use it to filter out multisomas.
            root_id_counts = nucleus_merged.groupby(['pt_root_id'])['id'].transform('count')
            nucleus_merged = nucleus_merged[root_id_counts == 1] 

            return nucleus_merged, segments

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Required data file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing nucleus data: {e}")