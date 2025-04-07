from caveclient import CAVEclient
from standard_transform import minnie_ds 

import numpy as np
import pandas as pd
import time as time
import requests
import os
from tqdm import tqdm


def download_nucleus_data(client, path2download, tables2download):
    """
    Download all the nucleus tables for further processing.

    Parameters:
        client: the CAVEClient used to download
        path2download: the location of the folder to download all this information
        tables2download: the list of tables to download
    """

    for table in tables2download:
        print(table)
        auxtable = client.materialize.query_table(table, split_positions=True)
        auxtable = pd.DataFrame(auxtable)
        auxtable.to_csv(f"{path2download}/{table}.csv")


def merge_nucleus_with_cell_types(nucleus_df, cell_type_df):
    """
    Merges nucleus data with cell types
        
    Returns:
        DataFrame merged with information about cell types
    """

    merged = nucleus_df.merge(cell_type_df, left_on=['id'], right_on=['target_id'], how='inner')
    merged = merged[['id_x', 'pt_root_id_x', 'pt_position_x_x',  'pt_position_y_x', 'pt_position_z_x', 'classification_system', 'cell_type']]
    return merged.rename(columns = {'id_x' : 'id', 'pt_root_id_x' : 'pt_root_id', 'pt_position_x_x' : 'pt_position_x','pt_position_y_x' : 'pt_position_y','pt_position_z_x' : 'pt_position_z'  }) 
    
    
def transform_positions(nucleus_df):
    """
    Transforms nuclei positions from voxels to μm
    
    Returns:
        DataFrame with the transformed positions
    """
    
    transformed_positions = np.empty((len(nucleus_df), 3)) 
    
    k = 0
    for (x,y,z) in tqdm(nucleus_df[['pt_position_x', 'pt_position_y','pt_position_z']].values, desc="Transform positions"):
    #for x,y,z in nucleus_df[['pt_position_x', 'pt_position_y','pt_position_z']].values:
        position = np.array([x,y,z])
        transformed = minnie_ds.transform_vx.apply(position)
        transformed_positions[k, :] = transformed
        k += 1
    
    #nucleus_df['position_um'] = transformed_positions
    
    nucleus_df['pt_position_x'] = transformed_positions[:, 0]#[pos[0] for pos in nucleus_df['position_um']]
    nucleus_df['pt_position_y'] = transformed_positions[:, 1]#[pos[1] for pos in nucleus_df['position_um']]
    nucleus_df['pt_position_z'] = transformed_positions[:, 2]#[pos[2] for pos in nucleus_df['position_um']]
    
    return nucleus_df


## Adding layer segmentation and assegnations
LAYER_CELL_TYPES = {
    'L1': ['NGC', 'BPC', 'MC', 'BC'],
    'L2/3': ['23P'],
    'L4': ['4P'],
    'L5': ['5P-IT', '5P-ET', '5P-NP'],
    'L6': ['6P-IT', '6P-CT'],
    'WM': ['Oligo', 'OPC', 'Pericyte']
}

LAYER_ORDER = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'WM']


def divide_volume_into_segments(cells_df, segment_size=10.0):
    """
    Divide the volume into segments along the y-axis.
    
    Parameters:
        cells_df (pd.DataFrame): DataFrame with cell information
        segment_size (float): Size of each segment in micrometers
    
    Returns:
        pd.DataFrame: Segmented layer information
    """
    # Calculate the number of segments
    y_min, y_max = cells_df['pt_position_y'].min(), cells_df['pt_position_y'].max()
    num_segments = int(np.ceil((y_max - y_min) / segment_size))
    
    # Create bins for segmentation
    y_bins = np.linspace(y_min, y_max, num_segments + 1)
    
    segments = []
    
    l23_assigned = False
    
    # Process each segment
    for i in range(len(y_bins) - 1):
        y_start, y_end = y_bins[i], y_bins[i + 1]
        y_center = (y_start + y_end) / 2
        
        segment_cells = cells_df[(cells_df['pt_position_y'] >= y_start) & (cells_df['pt_position_y'] < y_end)]
        
        layer_counts = {}
        for layer_name, cell_types in LAYER_CELL_TYPES.items():
            layer_cells = segment_cells[segment_cells['cell_type'].isin(cell_types)]
            layer_counts[layer_name] = len(layer_cells)
        
        # Special logic for L1 and L2/3
        if not l23_assigned:
            # Count L2/3 cells
            l23_cells = layer_counts.get('L2/3', 0)
            
            # If L2/3 cells are less than threshold, assign L1
            if l23_cells < 300:
                dominant_layer = 'L1'
            else:
                # Once we exceed the threshold, assign L2/3 and set the flag
                dominant_layer = 'L2/3'
                l23_assigned = True
        elif any(layer_counts.values()):
            dominant_layer = max(layer_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant_layer = 'Unknown'
        
        segments.append({
            'y_start': y_start,
            'y_end': y_end,
            'y_center': y_center,
            'L1_cells': layer_counts.get('L1', 0),
            'L2/3_cells': layer_counts.get('L2/3', 0),
            'L4_cells': layer_counts.get('L4', 0),
            'L5_cells': layer_counts.get('L5', 0),
            'L6_cells': layer_counts.get('L6', 0),
            'WM_cells': layer_counts.get('WM', 0),
            'dominant_layer': dominant_layer
        })
    
    segments_df = pd.DataFrame(segments)
    
    segments_df = enforce_layer_order(segments_df)
    
    return segments_df

def enforce_layer_order(segments_df):
    """
    Enforce correct anatomical order of layers.
    
    Parameters:
        segments_df (pd.DataFrame): DataFrame of segments
    
    Returns:
        pd.DataFrame: Corrected segments DataFrame
    """
    corrected_df = segments_df.copy()
    
    corrected_df['layer_index'] = corrected_df['dominant_layer'].apply(
        lambda x: LAYER_ORDER.index(x) if x in LAYER_ORDER else -1
    )
    
    last_valid_index = -1
    
    for i in range(len(corrected_df)):
        current_idx = corrected_df.iloc[i]['layer_index']
        
        if current_idx == -1:
            continue
        
        if last_valid_index == -1:
            last_valid_index = current_idx
            continue
        
        if current_idx < last_valid_index - 1:
            corrected_df.at[i, 'layer_index'] = last_valid_index
            current_idx = last_valid_index
        
        elif current_idx > last_valid_index + 1:
            corrected_df.at[i, 'layer_index'] = last_valid_index + 1
            current_idx = last_valid_index + 1
        
        last_valid_index = current_idx
    
    corrected_df['dominant_layer'] = corrected_df['layer_index'].apply(
        lambda x: LAYER_ORDER[x] if x >= 0 and x < len(LAYER_ORDER) else 'Unknown'
    )
    
    corrected_df = corrected_df.drop('layer_index', axis=1)
    
    return corrected_df

def merge_segments_by_layer(segments_df):
    """
    Merge segments that belong to the same layer.
    
    Parameters:
    -----------
    segments_df : pandas.DataFrame
        DataFrame of segments to be merged
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with merged layer segments
    """
    layer_groups = {}
    current_layer = None
    start_idx = 0
    
    for i, row in segments_df.iterrows():
        layer = row['dominant_layer']
        
        if layer == 'Unknown' or layer == 'WM':  # Skip Unknown and White Matter
            continue
        
        if current_layer != layer:
            if current_layer is not None:
                if current_layer not in layer_groups:
                    layer_groups[current_layer] = []
                
                layer_groups[current_layer].append({
                    'y_start': segments_df.iloc[start_idx]['y_start'],
                    'y_end': segments_df.iloc[i-1]['y_end']
                })
            
            current_layer = layer
            start_idx = i
    
    # Handle the last segment
    if current_layer is not None and start_idx < len(segments_df):
        if current_layer not in layer_groups:
            layer_groups[current_layer] = []
        
        layer_groups[current_layer].append({
            'y_start': segments_df.iloc[start_idx]['y_start'],
            'y_end': segments_df.iloc[len(segments_df)-1]['y_end']
        })
    
    merged_layers = []
    for layer, regions in layer_groups.items():
        for i, region in enumerate(regions):
            merged_layers.append({
                'layer': layer,
                'region_index': i,
                'y_start': region['y_start'],
                'y_end': region['y_end'],
                'height': region['y_end'] - region['y_start']
            })
    
    merged_df = pd.DataFrame(merged_layers)
    
    return merged_df.sort_values('y_start')


def connectome_constructor(client, presynaptic_set, postsynaptic_set, savefolder, neurs_per_steps = 500, start_index=0, max_retries=10, delay=5, drop_synapses_duplicates=True):
    '''
    Function to construct the connectome subset for the neurons specified in the presynaptic_set and postsynaptic_set.

    Args:
    client: CAVEclient needed to access MICrONS connectomics data
    presynaptic_set: 1-d array of non repeated root_ids of presynaptic neurons for which to extract postsynaptoc connections in postynaptic_set
    postynaptic_set: 1-d array of non repeated root_ids of postsynaptic neurons for which to extract presynaptic connections in presynaptic_set
    neurs_per_steps: number of postsynaptic neurons for which to recover presynaptic connectivity per single call to the connectomics
        database. Since the connectomics database has a limit on the number of connections you can query at once
        this iterative method optimises querying multiple neurons at once, as opposed to each single neuron individually,
        while also preventing the queries from crashing. I have tested that for a presynaptic set of around 8000 neurons
        you can reliably extract the connectivity for around 500 postsynaptic neurons at a time.
    '''
    
    #We are doing the neurons in packages of neurs_per_steps. If neurs_per_steps is not
    #a divisor of the postsynaptic_set the last iteration has less neurons 
    n_before_last = (postsynaptic_set.size//neurs_per_steps)*neurs_per_steps
    
    #Time before starting the party
    time_0 = time.time() 

    synapse_table = client.info.get_datastack_info()["synapse_table"]

    #Preset the dictionary so we do not build a large object every time
    syndfs = []
    neurons_to_download = {"pre_pt_root_id":presynaptic_set}
    part = start_index
    for i in range(start_index*neurs_per_steps, postsynaptic_set.size, neurs_per_steps):
        #Inform about our progress
        print(f"Postsynaptic neurons queried so far: {i}...")

        cols_2_download = ["pre_pt_root_id", "post_pt_root_id", "size", "ctr_pt_position"]
        #Try to query the API several times
        for retry in range(max_retries):
            try:
                #Get the postids that we will be grabbing in this query. We will get neurs_per_step of them
                post_ids = postsynaptic_set[i:i+neurs_per_steps] if i < n_before_last else postsynaptic_set[i:]
                neurons_to_download["post_pt_root_id"] = post_ids

                #Query the table 
                sub_syn_df = client.materialize.query_table(synapse_table,
                                                    filter_in_dict=neurons_to_download,
                                                    select_columns=cols_2_download,
                                                    split_positions=True)
                


                #Sum all repeated synapses. The last reset_index is because groupby would otherwise create a 
                #multiindex dataframe and we want to have pre_root and post_root as columns
                if drop_synapses_duplicates:
                    sub_syn_df = sub_syn_df.groupby(["pre_pt_root_id", "post_pt_root_id"]).sum().reset_index()

                sub_syn_df.to_csv(f'{savefolder}/connections_table_{part}.csv', index = False)
                part += 1

                #Add the result to the table
                #syndfs.append(sub_syn_df.values)

                #Measure how much time in total our program did run
                elapsed_time = time.time() - time_0

                #Use it to give the user an estimation of the end time.
                neurons_done = i+neurs_per_steps
                time_per_neuron = elapsed_time / neurons_done  
                neurons_2_do = postsynaptic_set.size - neurons_done
                remaining_time = time_format(neurons_2_do * time_per_neuron) 
                print(f"Estimated remaining time: {remaining_time}")
                break
            #If it a problem of the client, just retry again after a few seconds
            except requests.HTTPError as excep: 
                print(f"API error. Retry in {delay} seconds...")
                print(excep)
                time.sleep(delay)
                print(f"Trial {retry} failed. Resuming operations...")
                continue
            #If not, just raise the exception and that's all
            except Exception as excep: 
                raise excep

        #If the above loop did not succeed for any reason, then just abort.
        if retry >= max_retries:
            raise TimeoutError("Exceeded the max_tries when trying to get synaptic connectivity")
    
    #syn_df = pd.DataFrame({'pre_id':np.vstack(syndfs)[:, 0], 'post_id': np.vstack(syndfs)[:, 1], 'syn_volume': np.vstack(syndfs)[:, 2]})
    #return syn_df
    return

def time_format(seconds):
    if seconds > 3600*24: 
        days = int(seconds//(24*3600))
        hours = int((seconds - days*24*3600)//3600)
        return f"{days} days, {hours}h"
    elif seconds > 3600: 
        hours = int(seconds//3600)
        minutes = int((seconds - hours*3600) // 60)
        return f"{hours}h, {minutes}min"
    elif seconds > 60:
        minutes = int(seconds//60)
        rem_sec = int((seconds - 60*minutes))
        return f"{minutes}min {rem_sec}s"
    else:
        return f"{seconds:.0f}s"


def merge_connection_tables(savefolder):
    #Count the number of tables to merge, by checking all files in the correct folder
    ntables = 0
    for file in os.listdir(f"{savefolder}/synapses/"):
        if os.path.isfile(f"{savefolder}/synapses/{file}"):
            if "connections_table_" in file:
                ntables += 1

    #Merge all of them
    table = pd.read_csv(f"{savefolder}/synapses/connections_table_0.csv")
    ntables
    for i in range(1, ntables):
        table = pd.concat([table, pd.read_csv(f"{savefolder}/synapses/connections_table_{i}.csv")])

    table.to_csv(f"{savefolder}/synapses.csv")
    return 