import os
import logging
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
    Supported 
    """
    SUPPORTED_VERSIONS = [1078, 1181, 1300, 1412, 1507]


    def __init__(self, datadir="data", version=1300, download_policy='minimum', extra_tables=[]):
        """
        Initialize the class and makes sure subfolders to download exist. Configures the tables to be downloaded (except synapses) via
        a download policy.

        Parameters:
        -----------
            datadir: string, optional
                Defaults to 'data'. Points to the folder where information will be downloaded.
            custom_tables: dict, optional 
                Used to override the default tables used to construct the unit table in a given version. The keys for the tables
                to be overrided are 'celltype' for the nucleus classification scheme, 'proofreading' for the prooreading table,
                'brain_areas' for assigned brain areas, 'func_props' for functional properties, and 'coreg' for the coregistration table.
            download_policy: str, optional
                Used to set how the tables should be downloaded. 'minimum' (the default) only downloads the minimum amount of tables necessary 
                to construct our unit table. 'all' gets all of them. 'extra' gets the same as 'minimum' plus the tables specified in `extra_tables`.
            extra_tables: list, optional
                List of extra table names to be downloaded. See the download_police for more information.

        Returns:
        --------
            None.
                This method is a constructor and does not return any value.
        """
        
        self.version = version
        self.datadir = datadir
        self.data_storage = f"{self.homedir}/{self.datadir}/{self.version}"

        logging.info(f"Data will be stored in: {self.data_storage}")

        # Ensure directories exist
        os.makedirs(self.data_storage, exist_ok=True)
        os.makedirs(f"{self.data_storage}/raw", exist_ok=True)
        os.makedirs(f"{self.data_storage}/raw/synapses", exist_ok=True)
        logging.info("Data directories created successfully.")

        # Initialize the CAVEClient with the user-specified version 
        self._initialize_client(version)

        # Set the tables to download according to the version
        self._configure_download_tables(version, download_policy, extra_tables)

    
    def _configure_download_tables(self, version, download_policy, extra_tables):
        """
        Configures the list of tables to be downloaded based on version and policy.
        This is an internal helper method called by the constructor.
        This method is not intended for direct use by the end-user.
    
        Parameters:
        -----------
            version: int
                The dataset version, used to select the correct set of default table names.
            download_policy : {'minimum', 'all', 'extra', 'custom'}
                The policy that determines which tables will be included in the download list.
            extra_tables : list[str]
                A list of extra table names to be considered when the `download_policy`
                is 'extra' or 'custom'.
    
        Returns:
        --------
            None.
                This method does not return a value. It modifies the instance attributes
                `self.tables` and `self.tables_2_download` in-place.
        """
        
        self.tables = {}

        self.tables['nucleus']      = "nucleus_detection_v0"
        self.tables['proofreading'] = "proofreading_status_and_strategy"
        self.tables['brain_areas']  = "nucleus_functional_area_assignment"

        match version:
            case 1078 | 1181 | 1300 | 1412:
                self.tables['celltype']     = "aibs_metamodel_celltypes_v661"
                self.tables['func_props']   = "functional_properties_v3_bcm"
                self.tables['coreg']        = "coregistration_manual_v4"
            case 1507: 
                self.tables['celltype']     = "aibs_metamodel_celltypes_v661"
                self.tables['func_props']   = "digital_twin_properties_bcm_coreg_v4"
                self.tables['coreg']        = "coregistration_manual_v4"
            case _:
                raise ValueError(f"The selected version is not supported. Please use one of the following: {self.SUPPORTED_VERSIONS}")


        # Set the tables that we will need to download.
        match download_policy:
            # Default. Gets only the ones we will use to generate our unit table 
            case 'minimum':
                self.tables_2_download = list(self.tables.values()) 

            # All gets all available tables for this version
            case 'all':
                self.tables_2_download = self.get_table_list()

            # Get the minimum ones + the tables specified in the extra array
            case 'extra':
                self.tables_2_download = list(self.tables.values()) + extra_tables 
            
            # Any other string is an error
            case _: 
                logging.error(f"Invalid download policy: {download_policy}")
                raise ValueError("`download_tables` must be either `default`, `all` or `extra`")

        # Eliminate any 'None' value that could have appeared
        self.tables_2_download = [x for x in self.tables_2_download if x is not None] 

    
    def download_functional_fits(self, foldername="functional"):
        """
        Downloads functional tuning curves and fits from Zenodo.
        This convenience method downloads a pre-processed data file containing
        functional tuning curves and their fits. The data is hosted on Zenodo
        and has been specifically prepared for use with this package. The method
        handles the creation of the destination folder before initiating the download.
        
        Parameters:
        -----------
        - **foldername : *str, optional***

            The name of the subfolder within the main data directory where the
            data will be saved, by default "functional". If this folder does
            not exist, it will be created.
        
        Returns:
        --------
        - **None.**

            This method does not return a value. It saves the downloaded data to
            a file named 'tuning_curves_fitted_v1.csv' inside the specified folder.

        """

        os.makedirs(f"{self.homedir}/{self.datadir}/{foldername}", exist_ok=True)
        down.download_functional_fits(f"{self.homedir}/{self.datadir}/{foldername}/tuning_curves_fitted_v1.csv")


    def _initialize_client(self, version):
        """
        Initializes and configures the CAVEClient for a specific dataset version.
        This internal helper method creates an instance of the CAVEclient,
        pointing to the 'minnie65_public' datastack, and sets the desired
        materialization version. It is called by the constructor and is not
        intended for direct use by the end-user.
        
        Parameters:
        -----------
            version: int
                The materialization version to be used for all subsequent data queries.
                If left at None, it points to the last one.
        
        Returns:
        --------
            None.
                This method does not return a value. It sets the `self.client`
                attribute on the class instance.
        """
        
        logging.debug(f"Initializing CAVEclient for minnie65_public, version {version}.")
        try:
            self.client = CAVEclient('minnie65_public') 
            self.client.version = version
            logging.debug("CAVEclient initialized successfully.")
        except requests.HTTPError as excep:
            if '503' in str(excep):
                logging.error("HTTP error 503: the MICrONS server is temporarily unavailable. Client cannot be used for new downloads.")
                print("HTTP error 503: the MICrONS server is temporarily unavailable. Client cannot be used for new downloads.")
            else:
                logging.error("Unhandled exception during while setting up the client: " + str(excep))
                raise excep
        return

    
    def get_table_list(self):
        """
        Retrieves a list of all available tables for the current version.
        This method queries the CAVEclient to get the names of all tables
        available in the currently configured materialization version.
        
        Parameters:
        -----------
            None.
            
        Returns:
        --------
            list[str]
                A list of strings, where each string is the name of an available
                table in the dataset.
        """

        return self.client.materialize.get_tables()

    
    def read_table(self, table_name):
        """
        Reads a specified table from a local CSV file into a pandas DataFrame.
        
        Parameters:
        -----------
            table_name: str
                The name of the table to read.

        Returns:
        --------
            pandas.DataFrame
                A DataFrame containing the data from the specified CSV file.
        """

        if not table_name.endswith(".csv"):
            table_name += ".csv"
        
        return pd.read_csv(f"{self.data_storage}/raw/{table_name}") 

    
    def download_nucleus_data(self):
        """
        Downloads all nucleus-related tables based on the initial configuration.
        The specific tables downloaded depend on the `download_policy` chosen at initialization.
        
        Parameters:
        -----------
            None.
            
        Returns:
        --------
            None.
                This method does not return a value. It saves the downloaded tables
                as CSV files in the `raw` data directory. 
        """

        logging.info(f"Downloading nucleus data tables: {self.tables_2_download}")
        down.download_tables(self.client,f"{self.data_storage}/raw/",  self.tables_2_download)
        logging.info("Nucleus data download completed.")
        return

    
    def download_tables(self, table_names):
        """
        Downloads a user-specified list of tables.
        This method allows for downloading an arbitrary list of tables from the
        database, in addition to those specified by the download policy during
        initialization.
        
        Parameters:
        ----------- 
            table_names: list[str]
                A list containing the exact names of the tables to be downloaded.
        
        Returns:
        --------
            None.
                This method does not return a value. It saves the downloaded tables
                as CSV files in the `raw` data directory.
        """

        logging.info(f"Downloading custom tables: {table_names}")
        down.download_tables(self.client,f"{self.data_storage}/raw/",  table_names) 
        logging.info("Custom tables download completed.")
        return


    def download_synapse_data(self, presynaptic_set, postsynaptic_set, neurs_per_steps = 500, start_index=0, max_retries=10, delay=5, drop_synapses_duplicates=True):
        """
        Downloads synaptic connections between specified sets of neurons.
        
        Parameters:
        -----------
            presynaptic_set: numpy.ndarray
                A 1D NumPy array of unique `root_ids` for the presynaptic neurons.
            postsynaptic_set: numpy.ndarray
                A 1D NumPy array of unique `root_ids` for the postsynaptic neurons.
            neurs_per_steps: int, optional
                The number of postsynaptic neurons to query per batch, by default 500.
                This is crucial for managing API query size and preventing crashes.
            start_index: int, optional
                The batch index from which to start or resume the download, by default 0.
                Set this to continue an interrupted download.
            max_retries: int, optional
                The maximum number of times to retry a failed API query before
                raising an error, by default 10.
            delay: int, optional
                The number of seconds to wait between retries if an API error occurs,
                by default 5.
            drop_synapses_duplicates: bool, optional
                If True (default), all synapses between any two neurons are merged into a
                single entry, with `synapse_size` being the sum of individual sizes.
                If False, each synapse is kept as a separate record.
        
        Returns:
        --------
            None.
                This method does not return a value. It saves the downloaded synapse
                tables as a series of CSV files in the `raw/synapses` directory.
        """
       
        logging.debug("Starting synapse data download.")
        down.connectome_constructor(self.client, presynaptic_set, postsynaptic_set, f"{self.data_storage}/raw/synapses",
                                   neurs_per_steps = neurs_per_steps, start_index=start_index, max_retries=max_retries, delay=delay, drop_synapses_duplicates=drop_synapses_duplicates)
        logging.debug("Synapse data download completed.")
        return

    
    def merge_synapses(self, syn_table_name):
        """
        Merges downloaded synapse data batches into a single CSV file.

        Parameters:
        -----------
            syn_table_name: str
                The name for the output file that will contain the merged synapse data.
                
        Returns:
        --------
            None.
                This method does not return a value. It saves the merged table to a file.
        """
        
        down.merge_connection_tables(f"{self.data_storage}/raw", syn_table_name)
        return

    
    def merge_table(self, unit_table, new_table, columns, method="nucleus_id", how='left'):
        """
        Merges new columns from a source table into the main unit table.

        Parameters:
        -----------
            unit_table: pandas.DataFrame
                The primary DataFrame to which new columns will be added.
            new_table: pandas.DataFrame
                The source DataFrame from which to pull the new columns.
            columns: list[str] or None
                A list of column names from `new_table` to merge into `unit_table`.
                If None, all columns from `new_table` (except the join keys) are used.
            method: str 
                How the tables will be compared to each other. If 'nucleus_id' (default), the target_id 
                is matched to the nucleus_id. If functional, the session, scan and unit_id are compared. 
                If 'pt_root_id', merge by 'pt_root_id'. This last option is not adviced, unless is the only index available.
            how: str
                Equivalent to Panda's how argument for the merge function. Only 'inner' or 'left' are allowed, since the 
                new columns are always added into the nucleus table.
        
        Returns:
        --------
            pandas.DataFrame
                The merged DataFrame with the newly added columns.
        """
        
        return proc.merge_columns(unit_table, new_table, columns, method=method, how=how)

    
    def process_nucleus_data(self, functional_data='none'):
        """
        Processes downloaded data to generate a final, annotated unit table.
        It reads all previously downloaded nucleus-related tables (cell type,
        proofreading, brain area, etc.), merges them sequentially, transforms
        coordinates, performs cortical layer segmentation, and optionally
        integrates functional data based on the chosen strategy.
        
        Parameters:
        -----------
            functional_data: {'none', 'match', 'all', 'best_only'}, optional
                Specifies how to integrate functional data, by default 'none'.
                - 'none': No functional data is added.
                - 'match': Adds columns (`session`, `scan_idx`, `functional_unit_id`)
                  to allow matching units with their corresponding functional scans.
                - 'all': Merges functional data from all available scans for each unit,
                  potentially resulting in multiple rows per unit.
                - 'best_only': Merges data only from the scan with the highest
                  performance metric (`cc_abs`) for each unit.
        
        Returns:
        --------
            nucleus_merged: pandas.DataFrame
                The primary, processed unit table with all annotations, including
                cell type, proofreading status, brain area, cortical layer, and
                optionally functional properties.
            segments: pandas.DataFrame
                A DataFrame detailing the calculated cortical layer segments,
                including their start and end coordinates.
        """
        
        logging.info(f"Processing nucleus data with functional data option: '{functional_data}'.")
        try:

            # Read all the downloaded data
            logging.debug("Reading downloaded data files.")
            nucleus   = pd.read_csv(f"{self.data_storage}/raw/{self.tables['nucleus']}.csv")
            celltype  = pd.read_csv(f"{self.data_storage}/raw/{self.tables['celltype']}.csv")
            proofread = pd.read_csv(f"{self.data_storage}/raw/{self.tables['proofreading']}.csv")
            areas     = pd.read_csv(f"{self.data_storage}/raw/{self.tables['brain_areas']}.csv")

            # Use a better index for the global id
            nucleus = nucleus.rename(columns={'id' : 'nucleus_id'})

            # Load functional data depending on user preferences
            if functional_data in ['best_only', 'all']: 
                funcprops = pd.read_csv(f"{self.data_storage}/raw/{self.tables['func_props']}.csv")
            elif functional_data == 'match':
                coreg = pd.read_csv(f"{self.data_storage}/raw/{self.tables['coreg']}.csv")

            # Call all the merge functions. First, cell types
            logging.debug("Merging nucleus data with cell types.")
            nucleus_merged = proc.merge_nucleus_with_cell_types(nucleus, celltype)

            # Then, brain area. 
            logging.debug("Merging brain area information.")
            nucleus_merged = proc.merge_brain_area(nucleus_merged, areas)

            # Proofreading info
            logging.debug("Merging proofreading status.")
            nucleus_merged = proc.merge_proofreading_status(nucleus_merged, proofread, self.version)

            # Get the correct positions
            logging.debug("Transforming positions.")
            nucleus_merged = proc.transform_positions(nucleus_merged)

            # Segment the data and add the information about layers
            logging.debug("Segmenting volume and adding layer info.")
            segments = proc.divide_volume_into_segments(nucleus_merged)
            segments = proc.merge_segments_by_layer(segments)

            proc.add_layer_info(nucleus_merged, segments)

            # Clean the resulting table by eliminating all multisoma objects. 
            logging.debug("Cleaning table: removing multisoma objects and duplicates.")
            nucleus_merged = nucleus_merged[nucleus_merged['pt_root_id'] > 0]
            nucleus_merged = nucleus_merged.drop_duplicates(subset='pt_root_id', keep=False)


            # Finally, functional properties. 
            logging.debug("Adding functional information")
            if functional_data in ['best_only', 'all']: 
                nucleus_merged = proc.merge_functional_properties(nucleus_merged, funcprops, mode=functional_data)
                nucleus_merged.loc[nucleus_merged['tuning_type'].isna(), 'tuning_type'] = 'not_matched'
            elif functional_data == 'match':
                nucleus_merged = proc.merge_functional_properties(nucleus_merged, coreg, mode=functional_data)
                nucleus_merged.loc[nucleus_merged['tuning_type'].isna(), 'tuning_type'] = 'not_matched'
            else:
                nucleus_merged['tuning_type'] = 'not_matched'

            return nucleus_merged, segments

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Required data file not found: {e}")
        except Exception as e:
            raise e 
