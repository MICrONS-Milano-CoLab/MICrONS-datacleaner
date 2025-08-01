import pandas as pd
import time as time 
import requests
import os
import logging
from tqdm import tqdm

def download_tables(client, path2download, tables2download):
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

	# Download all the tables in the list
	for table in tqdm(tables2download, "Downloading nucleus tables..."):
		try:
			auxtable = client.materialize.query_table(table, split_positions=True)
			auxtable = pd.DataFrame(auxtable)
			auxtable.to_csv(f'{path2download}/{table}.csv', index=False)
		except Exception as e:
			raise RuntimeError(f'Error downloading table {table}: {e}')


def connectome_constructor(
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
		logging.info("Dropping synapse duplicates and excluding position data for lighter queries.")	
	else:
		cols_2_download = ['pre_pt_root_id', 'post_pt_root_id', 'size', 'ctr_pt_position']
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

					sub_syn_df.to_csv(f'{savefolder}/connections_table_{part}.csv', index=False)
					logging.info(f"Successfully saved connections_table_{part}.csv")				
					part += 1

					# Measure how much time in total our program did run
					elapsed_time = time.time() - time_0
					neurons_done = min(i + neurs_per_steps, postsynaptic_set.size)
					time_per_neuron = elapsed_time / neurons_done
					neurons_2_do = postsynaptic_set.size - neurons_done
					remaining_time = time_format(neurons_2_do * time_per_neuron)
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


def time_format(seconds):
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


def merge_connection_tables(savefolder, filename):
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


def download_functional_fits(filepath):
	"""
 	Downloads functional fit data from a static Zenodo repository.
    This function retrieves a CSV file containing functional fitting data from a
    pre-defined Zenodo URL and saves it to the specified local path.
    
    Parameters:
    -----------
    	filepath: str
        	The full path, including the desired filename, where the downloaded file will be stored.
   
    Returns:
    --------
        None.
        	This function does not return a value. It saves the content directly to a file.
	"""

	# TODO
	response = requests.get("URL TO OUR FILE IN ZENODO")

	with open(f"{filepath}.csv", mode="wb") as file:
		file.write(response.content)
