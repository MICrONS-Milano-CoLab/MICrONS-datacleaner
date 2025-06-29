import pandas as pd
import time as time
import requests
import os


def download_nucleus_data(client, path2download, tables2download):
	"""
	Download all the nucleus tables for further processing.

	Parameters:
    -----------
	    client: the CAVEClient used to download
	    path2download: the location of the folder to download all this information
	    tables2download: the list of tables to download
    
	Returns:
	-------
        None. All results are downloaded into files
	"""

	# Ensure directory exists
	os.makedirs(path2download, exist_ok=True)

	for table in tables2download:
		print(table)
		try:
			auxtable = client.materialize.query_table(table, split_positions=True)
			auxtable = pd.DataFrame(auxtable)
			auxtable.to_csv(f'{path2download}/{table}.csv')
		except Exception as e:
			raise RuntimeError(f'Error downloading table {table}: {e}')


def connectome_constructor(
	client, presynaptic_set, postsynaptic_set, savefolder, neurs_per_steps=500, start_index=0, max_retries=10, delay=5, drop_synapses_duplicates=True
):
	"""
	Function to construct the connectome subset for the neurons specified in the presynaptic_set and postsynaptic_set.

	Parameters:
    ------------
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

	Returns:
	-------
        None. All results are downloaded into files
	"""

	# Ensure directory exists
	os.makedirs(savefolder, exist_ok=True)

	# We are doing the neurons in packages of neurs_per_steps. If neurs_per_steps is not
	# a divisor of the postsynaptic_set the last iteration has less neurons
	n_before_last = (postsynaptic_set.size // neurs_per_steps) * neurs_per_steps

	# Time before starting the party
	time_0 = time.time()

	synapse_table = client.info.get_datastack_info()['synapse_table']

	# Preset the dictionary so we do not build a large object every time
	neurons_to_download = {'pre_pt_root_id': presynaptic_set}

	# If we are not getting individual synapses, the best thing we can do is to not ask for positions, which is very heavy
	if drop_synapses_duplicates:
		cols_2_download = ['pre_pt_root_id', 'post_pt_root_id', 'size']
	else:
		cols_2_download = ['pre_pt_root_id', 'post_pt_root_id', 'size', 'ctr_pt_position']
	part = start_index
	for i in range(start_index * neurs_per_steps, postsynaptic_set.size, neurs_per_steps):
		# Inform about our progress
		print(f'Postsynaptic neurons queried so far: {i}...', flush=True)

		# Try to query the API several times
		success = False  # Flag to track if current batch succeeded
		retry = 0
		while retry < max_retries and not success:
			try:
				# Get the postids that we will be grabbing in this query. We will get neurs_per_step of them
				post_ids = postsynaptic_set[i : i + neurs_per_steps] if i < n_before_last else postsynaptic_set[i:]
				neurons_to_download['post_pt_root_id'] = post_ids

				# Query the table
				sub_syn_df = client.materialize.query_table(
					synapse_table, filter_in_dict=neurons_to_download, select_columns=cols_2_download, split_positions=True
				)

				# Sum all repeated synapses. The last reset_index is because groupby would otherwise create a
				# multiindex dataframe and we want to have pre_root and post_root as columns
				if drop_synapses_duplicates:
					sub_syn_df = sub_syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).sum().reset_index()

				sub_syn_df.to_csv(f'{savefolder}/connections_table_{part}.csv', index=False)
				part += 1

				# Measure how much time in total our program did run
				elapsed_time = time.time() - time_0

				# Use it to give the user an estimation of the end time.
				neurons_done = min(i + neurs_per_steps, postsynaptic_set.size)
				time_per_neuron = elapsed_time / neurons_done
				neurons_2_do = postsynaptic_set.size - neurons_done
				remaining_time = time_format(neurons_2_do * time_per_neuron)
				print(f'Estimated remaining time: {remaining_time}', flush=True)
				success = True  # Mark batch as successful
			# If it a problem of the client, just retry again after a few seconds
			except requests.HTTPError as excep:
				print(f'API error. Retry in {delay} seconds...')
				print(excep)
				time.sleep(delay)
				print(f'Trial {retry} failed. Resuming operations...')
				retry += 1
			# If not, just raise the exception and that's all
			except Exception as excep:
				raise excep

		# If the above loop did not succeed for any reason, then just abort
		if not success:
			raise TimeoutError('Exceeded the max_tries when trying to get synaptic connectivity')

	return


def time_format(seconds):
    """
	A function aimed to format the remaining download seconds nicely.

	Parameters:
	-----------
        seconds: remaining download time in seconds
	
	Returns:
	--------
        String with a well-formated time
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
	Merge connection tables that were saved by connectome_constructor.

	Parameters:
    ------------
	    savefolder: The folder containing the connection tables
	    filename: Name of the output merged file 

	Returns:
	---------
        Nothing. Merges everything to a file 
	"""
	# Check if the synapses folder exists
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
		print('No connection tables found to merge')
		return

	# Merge all of them
	first_file = connection_files[0]
	table = pd.read_csv(first_file)

	for file_path in connection_files[1:]:
		table = pd.concat([table, pd.read_csv(file_path)])

	output_path = f'{savefolder}/{filename}.csv'
	table.to_csv(output_path)
	print(f'Merged {len(connection_files)} tables into {output_path}')
	return
