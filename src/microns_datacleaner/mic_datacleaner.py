import os
import logging
from pathlib import Path
import pandas as pd

from caveclient import CAVEclient

from . import downloader as down
from . import processing as proc

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.info("Initializing MicronsDataCleaner...")
        self.version = version
        self.datadir = datadir
        self.data_storage = f"{self.homedir}/{self.datadir}/{self.version}"
        logging.info(f"Data will be stored in: {self.data_storage}")

        # Ensure directories exist
        os.makedirs(self.data_storage, exist_ok=True)
        os.makedirs(f"{self.data_storage}/raw", exist_ok=True)
        os.makedirs(f"{self.data_storage}/raw/synapses", exist_ok=True)
        logging.info("Data directories created successfully.")

        #Initialize the CAVEClient with the user-specified version 
        self._initialize_client(version)

        #Set the tables to download according to the version
        self._configure_download_tables(version, custom_tables, download_policy, extra_tables)
        logging.info("MicronsDataCleaner initialized successfully.")

    def _configure_download_tables(self, version, custom_tables, download_policy, extra_tables):
        """
        Internal function that configures the tables to be downloaded according to the selected version and download policy.
        See the constructor for information on custom_tables and possible policies. This function is not intended for the user. 
        """

        self.tables = {}

        match version:
            case 1300:
                self.tables['nucleus']      = "nucleus_detection_v0"
                self.tables['celltype']     = "aibs_metamodel_celltypes_v661"
                self.tables['proofreading'] = "proofreading_status_and_strategy"
                self.tables['brain_areas']  = "nucleus_functional_area_assignment"
                self.tables['func_props']   = "functional_properties_v3_bcm"
                self.tables['coreg']        = "coregistration_manual_v4"

        #Override all defaults with user-provided custom tables IF they were indicated 
        for key in custom_tables:
            self.tables[key] = custom_tables[key]

        #Set the tables that we will need to download.
        match download_policy:
            #Default. Gets only the ones we will use to generate our unit table 
            case 'minimum':
                self.tables_2_download = list(self.tables.values()) 

            #All gets all available tables for this version
            case 'all':
                self.tables_2_download = self.get_table_list()

            #Get the minimum ones + the tables specified in the extra array
            case 'extra':
                self.tables_2_download = list(self.tables.values()) + extra_tables 
            
            #Any other string is an error
            case _: 
                logging.error(f"Invalid download policy: {download_policy}")
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
        logging.info(f"Initializing CAVEclient for minnie65_public, version {version}.")
        try:
            self.client = CAVEclient('minnie65_public') 
            self.client.version = version
            logging.info("CAVEclient initialized successfully.")
        except requests.HTTPError as excep:
            if '503' in str(excep):
                logging.warning("HTTP error 503: the MICrONS server is temporarily unavailable. Client cannot be used for new downloads.")
            else:
                logging.error(f"HTTP error while initializing client: {excep}")
        return

    def get_table_list(self):
        """
        Returns a complete list of the CAVEClient available tables for the selected version
        """
        logging.info(f"Fetching available tables for version {self.version}.")
        return self.client.materialize.get_tables()

    def download_nucleus_data(self):
        """
        Downloads all the tables indicated in the cleaner.tables to download 
        """
	logging.info(f"Downloading nucleus data tables: {self.tables_2_download}")
        down.download_tables(self.client,f"{self.data_storage}/raw/",  self.tables_2_download)
	logging.info("Nucleus data download completed.")
        return

    def download_tables(self, table_names):
        """
        Downloads the specified tables.

        Parameters
        ----------
            table_names: list of str, the names of the tables to be downloaded
        """
	logging.info(f"Downloading custom tables: {custom_tables}")
        down.download_tables(self.client,f"{self.data_storage}/raw/",  table_names) 
	logging.info("Custom tables download completed.")
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
        logging.info("Starting synapse data download.")
        down.connectome_constructor(self.client, presynaptic_set, postsynaptic_set, f"{self.data_storage}/raw/synapses",
                                   neurs_per_steps = neurs_per_steps, start_index=start_index, max_retries=max_retries, delay=delay, drop_synapses_duplicates=drop_synapses_duplicates)
        logging.info("Synapse data download completed.")
        return

    def merge_synapses(self, syn_table_name):
        """
        Merges all the batches of the downloaded synapses. 
        """
        logging.info(f"Merging synapse tables into '{syn_table_name}'.")
        down.merge_connection_tables(f"{self.data_storage}/raw", syn_table_name)
        logging.info("Synapse tables merged successfully.")
        return

    def merge_table(self, unit_table, new_table, columns, method="nucleus_id", how='left'):
        return proc.merge_columns(unit_table, new_table, columns, method=method, how=how)
        


    def process_nucleus_data(self, functional_data='none'):
        """
        Processes all the downloaded nucleus data to generate a unified units_table. This includes information on 
        brain area, functional data, proofreading, as well as a layer segmentation.

        Parameters:
        -----------
            with_functional: string. 
            If 'none', no functional data is added to the table (default). 
            If 'match', the 'session', 'scan_idx' and 'unit_id' indices are added to match the units with the corresponding functional scans.
            Note that 'unit_id' is renamed to 'functional_unit_id' to avoid confusions. 
            If 'all', the functional data from the digital twin is added, conserving units with multiple scans and the three aforementioned indices. 
            If 'best_only' only the scan with the highest performance from the digital twin is considered (and the scan indices are not added).  
        
        Returns:
        --------
            nucleus_merged: a unit_table with all the nucleus information processed.
            segments: an array with the positions of all segments
        """
        logging.info(f"Processing nucleus data with functional data option: '{with_functional}'.")
        try:

            #Read all the downloaded data
            logging.info("Reading downloaded data files.")
            nucleus   = pd.read_csv(f"{self.data_storage}/raw/{self.tables['nucleus']}.csv")
            celltype  = pd.read_csv(f"{self.data_storage}/raw/{self.tables['celltype']}.csv")
            proofread = pd.read_csv(f"{self.data_storage}/raw/{self.tables['proofreading']}.csv")

            areas     = pd.read_csv(f"{self.data_storage}/raw/{self.tables['brain_areas']}.csv")


            nucleus = nucleus.rename(columns={'id' : 'nucleus_id'})

            if functional_data in ['best_only', 'all']: 
                funcprops = pd.read_csv(f"{self.data_storage}/raw/{self.tables['func_props']}.csv")
            elif functional_data == 'match':
                coreg = pd.read_csv(f"{self.data_storage}/raw/{self.tables['coreg']}.csv")

            logging.info("Merging nucleus data with cell types.")
            #Call all the merge functions. First, cell types
            nucleus_merged = proc.merge_nucleus_with_cell_types(nucleus, celltype)

            logging.info("Merging brain area information.")
            #Then, brain area. 
            nucleus_merged = proc.merge_brain_area(nucleus_merged, areas)

            logging.info("Merging proofreading status.")
            #Proofreading info
            nucleus_merged = proc.merge_proofreading_status(nucleus_merged, proofread, self.version)

            logging.info("Transforming positions.")
            #Get the correct positions
            nucleus_merged = proc.transform_positions(nucleus_merged)

            logging.info("Segmenting volume and adding layer info.")
            #Segment the data and add the information about layers
            segments = proc.divide_volume_into_segments(nucleus_merged)
            segments = proc.merge_segments_by_layer(segments)

            proc.add_layer_info(nucleus_merged, segments)

            logging.info("Cleaning table: removing multisoma objects and duplicates.")
            #Clean the resulting table by eliminating all multisoma objects. 
            nucleus_merged = nucleus_merged[nucleus_merged['pt_root_id'] > 0]
            nucleus_merged = nucleus_merged.drop_duplicates(subset='pt_root_id', keep=False)


            #Finally, functional properties. In some versions, these need to be added from the functional data 
            if functional_data in ['best_only', 'all']: 
                nucleus_merged = proc.merge_functional_properties(nucleus_merged, funcprops, mode=functional_data)
            elif functional_data == 'match':
                nucleus_merged = proc.merge_functional_properties(nucleus_merged, coreg, mode=functional_data)

            logging.info("Nucleus data processing completed successfully.")
            #nucleus_merged = nucleus_merged.rename(columns = {'id' : 'nucleus_id'}) 

            return nucleus_merged, segments

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Required data file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing nucleus data: {e}")
