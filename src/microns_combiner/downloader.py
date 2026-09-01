import pandas as pd
import time as time 
import requests
import os
from io import BytesIO
import logging
from cloudfiles import CloudFiles
from tqdm import tqdm

from . import filters

def _download_tables(client, path2download, tables2download):
    """
    Download all the indicated tables for further processing.

    Parameters:
    -----------
        client: caveclient.CAVEclient 
             The CAVEclient instance used to connect to and download from the data service.
        path2download: str
            The local file path to the directory where the downloaded tables will be saved as CSV files.
        tables2download: list[str]
            A list containing the names of the tables to be downloaded.
    
    Returns:
    -------
        None.
            This function does not return any value. It saves the downloaded tables as files in the 
            specified directory.
    """
    
    logging.info(f"Starting download of nucleus data to {path2download}.")
        # Ensure directory exists
    os.makedirs(path2download, exist_ok=True)

    # Ensure directory exists
    os.makedirs(path2download, exist_ok=True)

    #Is this version active on the API or has been archived?
    #Select the correct way of downloading
    if _is_archived(client):
        _static_table_download(client.version, path2download, tables2download)
    else:
        _api_table_download(client, path2download, tables2download) 


def _api_table_download(client, path2download, tables2download):
    """
    Downloads a table from the CAVE API. 

    Parameters:
    -----------
        version: int
            The version number of the dataset
        path2download: str
            The local file path to the directory where the downloaded tables will be saved as CSV files.
        tables2download: list[str]
            A list containing the names of the tables to be downloaded.
        base_path: str
            The location of the cloud bucket.
    
    Returns:
    --------
        None
    """

    # Download all the tables in the list
    for table in tqdm(tables2download, "Downloading nucleus tables..."):
        try:
            auxtable = client.materialize.query_table(table, split_positions=True)
            auxtable = pd.DataFrame(auxtable)
            auxtable.to_csv(f'{path2download}/{table}.csv', index=False)
        except Exception as e:
            raise RuntimeError(f'Error downloading table {table}: {e}')

def _static_table_download(version, path2download, tables2download, base_path="gs://mat_dbs/public/minnie65_phase3_v1"):
    """
    Downloads a table from the static bucket instead of the CAVE API. Useful for datasets which have been archived 
    to the static version.

    Parameters:
    -----------
        version: int
            The version number of the dataset
        path2download: str
            The local file path to the directory where the downloaded tables will be saved as CSV files.
        tables2download: list[str]
            A list containing the names of the tables to be downloaded.
        base_path: str
            The location of the cloud bucket.
    
    Returns:
    --------
        None
    """

    #Uses the example code from MICrONs for this.
    cf = CloudFiles(f"{base_path}/v{version}", use_https=True)


    for table in tqdm(tables2download, "Downloading nucleus tables..."):
        try:
            print(f"{table}_merged.csv.gz")
            df_data = cf.get(f"{table}_merged.csv.gz")
            header_data = cf.get(f"{table}_merged_header.csv")

            with BytesIO(header_data) as f:
                header = pd.read_csv(f, header=None, names=["column", "type"])
            columns = header["column"].tolist()

            with BytesIO(df_data) as f:
                df = pd.read_csv(f, compression="gzip", header=None)
                df.columns = columns
                df.to_csv(f'{path2download}/{table}.csv', index=False)
        except Exception as e:
            raise RuntimeError(f'Error downloading table {table}: {e}')

def _is_archived(client):
    """
    Checks if the current version has been archived or can be queried by CAVE.

    Parameters:
    -----------
        client: CAVEclient
            The instance of the CAVE client to be checked
    Returns:
    --------
        is_archived: bool
            True if the version has been archived, False if it can be queried.
    """
    return client.version not in client.materialize.get_versions()

def _download_file(url, destination, chunk_size=1024*1024):
    """
        Auxiliary function to download a large file without using it all into RAM,
        at chunk sizes 

        Parameters:
        -----------
            url: string
                The url of the file to be downloaded. The file will be written to a file.
            destination: string
                Path of the file that will be created. 
            chunk_size: int, optional
                Defaults to 1024*1024.
        Returns:
        --------
            None

    """

    #Avoid filling too much RAM, downloading little by little
    with requests.get(url, stream=True) as response:
        #Check if all good
        response.raise_for_status()

        #Total file size
        total = int(response.headers.get('content-length', 0))

        #Start writing the file by doing iterative queries to our file
        with open(destination, "wb") as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    return None 

def _connectome_constructor(
    client, presynaptic_set, postsynaptic_set, savefolder, neurs_per_steps=500, start_index=0, max_retries=10, delay=5, drop_synapses_duplicates=True
):
    """
    Constructs a connectome subset for specified pre- and postsynaptic neurons.
    This function queries the MICrONS connectomics database to extract synaptic
    connections between a defined set of presynaptic and postsynaptic neurons.
    
    Parameters:
    -----------
        client: caveclient.CAVEclient
            The CAVEclient instance used to access MICrONS connectomics data.
        presynaptic_set: numpy.ndarray
            A 1D NumPy array of unique `root_ids` for the presynaptic neurons.
        postsynaptic_set: numpy.ndarray
            A 1D NumPy array of unique `root_ids` for the postsynaptic neurons.
        savefolder: str
            The path to the directory where the output files will be saved.
        neurs_per_steps: int, optional
            Number of postsynaptic neurons to query per batch, by default 500.
            This parameter enables querying the database in iterative batches to
            work around API query size limits. A value of 500 is a reliable
            choice for a presynaptic set of approximately 8000 neurons.
        start_index: int, optional
            The starting batch index for the download, by default 0. If a previous
            download was interrupted, this can be set to the index of the last
            successfully downloaded file to resume the process.
        max_retries: int, optional
            The maximum number of times to retry a query if the server fails to
            respond, by default 10.
        drop_synapses_duplicates: bool, optional
            If True (default), all synapses between a given pair of neurons (i, j)
            are merged into a single entry. The `synapse_size` of this entry will be
            the sum of all individual synapse sizes. If False, each synapse is
            saved as a separate entry.
     
    Returns:
    --------
        None.
            This function does not return any value. The resulting connection tables
            are saved as individual CSV files in the specified `savefolder`.
    """
    
    # Ensure directory exists
    os.makedirs(savefolder, exist_ok=True)

    # We are doing the neurons in packages of neurs_per_steps. If neurs_per_steps is not
    # a divisor of the postsynaptic_set the last iteration has less neurons
    n_before_last = (postsynaptic_set.size // neurs_per_steps) * neurs_per_steps
    n_chunks = 1 + (postsynaptic_set.size // neurs_per_steps)

    # Time before starting the party
    time_0 = time.time()

    synapse_table = client.info.get_datastack_info()['synapse_table']

    # Preset the dictionary so we do not build a large object every time
    neurons_to_download = {'pre_pt_root_id': presynaptic_set}

    # If we are not getting individual synapses, the best thing we can do is to not ask for positions, which is very heavy
    if drop_synapses_duplicates:
        cols_2_download = ['pre_pt_root_id', 'post_pt_root_id', 'size']
        cols_2_select  = cols_2_download
        logging.info("Dropping synapse duplicates and excluding position data for lighter queries.")	
    else:
        cols_2_download = ['pre_pt_root_id', 'post_pt_root_id', 'size', 'ctr_pt_position']
        #Since we are splitting position, when we select the columns we need to have the full name
        cols_2_select = ['pre_pt_root_id', 'post_pt_root_id', 'size'] + [f'ctr_pt_position_{A}' for A in 'xyz'] 
    part = start_index

    # Progress bar over the amount of chunks to download
    with tqdm(total=n_chunks) as progress_bar:
        # Main loop over chunks
        for i in range(start_index * neurs_per_steps, postsynaptic_set.size, neurs_per_steps):
            # Inform about our progress
            logging.debug(f'Postsynaptic neurons queried so far: {i}...')

            # Try to query the API several times
            success = False  # Flag to track if current batch succeeded
            retry = 0
            while retry < max_retries and not success:
                try:
                    # Get the postids that we will be grabbing in this query. We will get neurs_per_step of them
                    post_ids = postsynaptic_set[i : i + neurs_per_steps] if i < n_before_last else postsynaptic_set[i:]
                    neurons_to_download['post_pt_root_id'] = post_ids
                    logging.debug(f"Querying batch starting at index {i} with {len(post_ids)} neurons.")
                    # Query the table
                    sub_syn_df = client.materialize.query_table(
                        synapse_table, filter_in_dict=neurons_to_download, select_columns=cols_2_download, split_positions=True
                    )

                    # Sum all repeated synapses. The last reset_index is because groupby would otherwise create a
                    # multiindex dataframe and we want to have pre_root and post_root as columns
                    if drop_synapses_duplicates:
                        sub_syn_df = sub_syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).sum().reset_index()

                    #Filter the columns we want. Sometimes there are linked columns that are also downloaded
                    sub_syn_df = sub_syn_df[cols_2_select]  

                    #Save this part
                    sub_syn_df.to_csv(f'{savefolder}/connections_table_{part}.csv', index=False)
                    logging.info(f"Successfully saved connections_table_{part}.csv")				
                    part += 1

                    # Measure how much time in total our program did run
                    elapsed_time = time.time() - time_0
                    neurons_done = min(i + neurs_per_steps, postsynaptic_set.size)
                    time_per_neuron = elapsed_time / neurons_done
                    neurons_2_do = postsynaptic_set.size - neurons_done
                    remaining_time = _time_format(neurons_2_do * time_per_neuron)
                    logging.debug(f'Estimated remaining time: {remaining_time}')
                    success = True

                    # Set that another chunk was downloaded
                    progress_bar.update(1)

                except requests.HTTPError as excep:
                    logging.warning(f'API error on trial {retry + 1}. Retrying in {delay} seconds... Details: {excep}')
                    print(f'API error on trial {retry + 1}. Retrying in {delay} seconds... Details: {excep}')
                    time.sleep(delay)
                    retry += 1

                except Exception as excep:
                    logging.error(f"An unexpected error occurred: {excep}")
                    raise excep

    if not success:
        logging.error('Exceeded the max retries when trying to get synaptic connectivity. Aborting.')
        raise TimeoutError('Exceeded the max_tries when trying to get synaptic connectivity')



def _download_static_synapses(client, savefolder):
    """
    Used to download the entire synapse table from a static version.

    Parameters:
    -----------
        version: int
            Version of the release manifest 
        savefolder: str
            Folder where the synapse table should be stored.

    Returns:
    --------
        None
    """

    #Use googleapis instead of cloud.goole as it allows for anonymous access to public files
    base_url = (
        f"https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v{client.version}" 
    )

    #Name of the tables we want to download
    data_name = "synapses_pni_2_v1_filtered_view.csv.gz"
    header_name = "synapses_pni_2_v1_filtered_view_header.csv"

    # Ensure directory exists
    os.makedirs(savefolder, exist_ok=True)

    data_path = f"{savefolder}/{data_name}" 
    header_path = f"{savefolder}/{header_name}" 

    # Download only if not present
    if not os.path.exists(data_path):
        _download_file(f"{base_url}/{data_name}", data_path)
        _download_file(f"{base_url}/{header_name}", header_path)


def _filter_static_synapses(savefolder, presynaptic_set, postsynaptic_set, filename, drop_synapses_duplicates=True, chunksize=500_000):
    """
    Used to filter the entire static synapse table to generate a small one that can be worked with.  

    Parameters:
    -----------
        savefolder: str
            Folder where the synapse table is stored.
        presynaptic_set: numpy.ndarray
            A 1D NumPy array of unique `root_ids` for the presynaptic neurons.
        postsynaptic_set: numpy.ndarray
            A 1D NumPy array of unique `root_ids` for the postsynaptic neurons.
        filename: str
            The name of the filtered table
        drop_synapses_duplicates: bool, optional
            If True (default), all synapses between a given pair of neurons (i, j)
            are merged into a single entry. The `synapse_size` of this entry will be
            the sum of all individual synapse sizes. If False, each synapse is
            saved as a separate entry.
        chunksize: int, optional
            The size of the chunks to read from disk, to avoid running of RAM. Should be reduced in systems
            with smaller RAM sizes.
    Returns:
    --------
        None
    """

    #Where the synapse tables are located
    data_path   = f"{savefolder}/synapses_pni_2_v1_filtered_view.csv.gz"
    header_path = f"{savefolder}/synapses_pni_2_v1_filtered_view_header.csv"

    #Name of the columns are stored in the header file
    headerdata = pd.read_csv(header_path, header=None, names=['column', 'type'])
    columns = headerdata['column'].tolist()

    #Get the columns we want to use
    if drop_synapses_duplicates:
        selected_cols = ['pre_pt_root_id', 'post_pt_root_id', 'size']
        logging.info("Dropping synapse duplicates and excluding position data for lighter queries.")	
    else:
        selected_cols = ['pre_pt_root_id', 'post_pt_root_id', 'size'] + [f'ctr_pt_position_{A}' for A in 'xyz'] 

    #Initialize result table
    result = pd.DataFrame(columns=selected_cols)
    list_chunks = []

    #Read table chunk by chunk

    for chunk in tqdm(pd.read_csv(data_path, header=None, names=columns, usecols=selected_cols, chunksize=chunksize), unit=' chunks', desc='Filtering synapse table...'):
        #Filter
        filtered_chunk = filters.synapses_by_id(chunk, presynaptic_set, postsynaptic_set) 
        #Do it for every chunk, to avoid large tables if many ids are queried
        if drop_synapses_duplicates:
            filtered_chunk = filtered_chunk.groupby(['pre_pt_root_id', 'post_pt_root_id']).sum().reset_index()
        #Put all the chunks together
        #result = pd.concat([result, filtered_chunk]) 
        list_chunks.append(filtered_chunk)

    #Finally, call concat on the list of DF to generate the final dataframe in a fast way
    result = pd.concat(list_chunks, ignore_index=True)

    #Some duplicates may remain since we were operating in chunks, so just filter again if needed
    if drop_synapses_duplicates:
        result = result.groupby(['pre_pt_root_id', 'post_pt_root_id']).sum().reset_index()

    output_path = f'{savefolder}/{filename}.csv'
    result.to_csv(output_path, index=False)
    logging.info(f'Filtere synapse table into {output_path}')

    
    


def _time_format(seconds):
    """
    Formats a duration in seconds into a human-readable string.
    
    Parameters:
    -----------
        seconds: float
        The total duration in seconds to be formatted.
  
    Returns:
    --------
        str
        A string representing the formatted duration.
    """
    
    if seconds > 3600 * 24:
        days = int(seconds // (24 * 3600))
        hours = int((seconds - days * 24 * 3600) // 3600)
        return f'{days} days, {hours}h'
    elif seconds > 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds - hours * 3600) // 60)
        return f'{hours}h, {minutes}min'
    elif seconds > 60:
        minutes = int(seconds // 60)
        rem_sec = int((seconds - 60 * minutes))
        return f'{minutes}min {rem_sec}s'
    else:
        return f'{seconds:.0f}s'


def _merge_connection_tables(savefolder, filename):
    """
    Merges individual connection table files into a single master file.
    This function scans a specified directory for connection table files
    (identified by the prefix 'connections_table_'), which are typically
    generated by the `connectome_constructor` function. It then concatenates
    them into a single pandas DataFrame and saves the result as a new CSV file.
 
    Parameters:
    -----------
        savefolder: str
            The path to the directory containing the connection table files to be merged.
        filename: str
            The base name for the output file. The merged table will be saved as 
         '{filename}.csv' in the `savefolder`.
   
    Returns:
    --------
        None.
            This function does not return a value. It saves the merged table to a CSV file.
    """
    
    # Check if the synapses folder exists
    logging.info(f"Starting to merge connection tables into {filename}.csv")
    synapses_path = f'{savefolder}/synapses/'
    if not os.path.exists(synapses_path):
        if os.path.exists(savefolder) and any('connections_table_' in f for f in os.listdir(savefolder)):
            synapses_path = savefolder
        else:
            raise FileNotFoundError(f'Could not find synapses directory at {synapses_path}')

    # Count the number of tables to merge, by checking all files in the correct folder
    connection_files = []
    for file in os.listdir(synapses_path):
        file_path = os.path.join(synapses_path, file)
        if os.path.isfile(file_path) and 'connections_table_' in file:
            connection_files.append(file_path)

    if not connection_files:
        logging.warning('No connection tables found to merge.')
        return

    logging.info(f"Found {len(connection_files)} connection tables to merge.")
    
    # Merge all of them
    first_file = connection_files[0]
    table = pd.read_csv(first_file)

    for file_path in connection_files[1:]:
        table = pd.concat([table, pd.read_csv(file_path)])

    output_path = f'{savefolder}/{filename}.csv'
    table.to_csv(output_path, index=False)
    logging.info(f'Merged {len(connection_files)} tables into {output_path}')
    return


def _download_functional_data(filepath, chunk_size=1024*1024):
    """
    Downloads functional data from a static repository.
    This function retrieves a H5 file containing functional data from a
    predefined URL
    
    Parameters:
    -----------
        filepath: str
            The full path, including the desired filename, where the downloaded file will be stored.
        chunk_size: int, optional
            Defaults to 1024*1024.
   
    Returns:
    --------
        None.
            This function does not return a value. It saves the content directly to a file.
    """

    url = "https://huggingface.co/datasets/NeuroBLab/MICrONS/resolve/main/microns.h5"
    _download_file(url, filepath, chunk_size)