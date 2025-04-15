import numpy as np
import pandas as pd

from standard_transform import minnie_ds 
from scipy.stats import circmean
from tqdm import tqdm


def merge_nucleus_with_cell_types(nucleus_df, cell_type_df):
    """
    Merges nucleus data with cell types
        
    Returns:
        DataFrame merged with information about cell types
    """
    if nucleus_df.empty or cell_type_df.empty:
        print("Warning: Empty dataframe provided to merge_nucleus_with_cell_types")
        return pd.DataFrame()
    
    merged = nucleus_df.merge(cell_type_df, left_on=['id'], right_on=['target_id'], how='inner')
    merged = merged[['id_x', 'pt_root_id_x', 'pt_position_x_x',  'pt_position_y_x', 'pt_position_z_x', 'classification_system', 'cell_type']]
    return merged.rename(columns = {'id_x' : 'id', 'pt_root_id_x' : 'pt_root_id', 'pt_position_x_x' : 'pt_position_x','pt_position_y_x' : 'pt_position_y','pt_position_z_x' : 'pt_position_z'  }) 
    
def merge_brain_area(nucleus_df, areas):
    """
    Merges nucleus data with brain area information
    
    Returns:
        DataFrame merged with brain area information
    """
    if nucleus_df.empty or areas.empty:
        print("Warning: Empty dataframe provided to merge_brain_area")
        return nucleus_df
    
    merged = nucleus_df.merge(areas, left_on=['id'], right_on=['target_id'], how='inner')
    merged = merged[['id_x', 'pt_root_id_x', 'pt_position_x_x',  'pt_position_y_x', 'pt_position_z_x', 
                     'classification_system', 'cell_type', 'tag']]

    return merged.rename(columns = {'id_x' : 'id', 'pt_root_id_x' : 'pt_root_id', 
                                    'pt_position_x_x' : 'pt_position_x','pt_position_y_x' : 'pt_position_y','pt_position_z_x' : 'pt_position_z',
                                    'tag' : 'brain_area'}) 

def merge_proofreading_status(nucleus_df, proofreading):
    """
    Merges nucleus data with proofreading status information
    
    Returns:
        DataFrame merged with proofreading information
    """
    if nucleus_df.empty:
        print("Warning: Empty nucleus dataframe provided to merge_proofreading_status")
        return nucleus_df
    
    merged = nucleus_df.merge(proofreading, left_on=['pt_root_id'], right_on=['pt_root_id'], how='left')
    merged = merged[['pt_root_id', 'id_x', 'pt_position_x_x',  'pt_position_y_x', 'pt_position_z_x', 
                     'classification_system', 'cell_type', 'brain_area', 'strategy_dendrite', 'strategy_axon']]

    #Tag the ones that have not been proofread
    merged.loc[merged['strategy_dendrite'].isna(), "strategy_dendrite"] = "none" 
    merged.loc[merged['strategy_axon'].isna(), "strategy_axon"] = "none" 

    return merged.rename(columns = {'id_x' : 'id', 'pt_position_x_x' : 'pt_position_x','pt_position_y_x' : 'pt_position_y',
                                    'pt_position_z_x' : 'pt_position_z'}) 

def merge_functional_properties(nucleus_df, functional, use_directions=False):
    """
    Merges nucleus data with functional properties
    
    Args:
        nucleus_df: DataFrame with nucleus information
        functional: DataFrame with functional properties
        use_directions: Whether to use directions for angle calculations (default: False)
    
    Returns:
        DataFrame merged with functional properties
    """
    if nucleus_df.empty or functional.empty:
        print("Warning: Empty dataframe provided to merge_functional_properties")
        return nucleus_df
    
    #Take all scans/sessions for each target_id, then average over them.
    #For the angles we need to use the circmean, so employ apply + a lambda function that returns the average of each thing separately
    high_circmean = 2*np.pi if use_directions else np.pi
    funcmean = functional.groupby(['target_id']).apply(lambda x: pd.Series({'pref_ori' : circmean(x['pref_ori'], low=0, high=high_circmean), 'gOSI': x['gOSI'].mean()})) 

    #Then proceed on the merge. In this case there is no common columns so filtering and renaming the result 
    #after the merge is not necessary anymore
    merged = nucleus_df.merge(funcmean, left_on=['id'], right_on=['target_id'], how='left')

    return merged
    
def transform_positions(nucleus_df):
    """
    Transforms nuclei positions from voxels to μm
    
    Returns:
        DataFrame with the transformed positions
    """
    if nucleus_df.empty:
        print("Warning: Empty dataframe provided to transform_positions")
        return nucleus_df
    
    transformed_positions = np.empty((len(nucleus_df), 3)) 
    
    k = 0
    for k, (x, y, z) in enumerate(tqdm(nucleus_df[['pt_position_x', 'pt_position_y', 'pt_position_z']].values, desc="Transform positions")):
        position = np.array([x,y,z])
        transformed = minnie_ds.transform_vx.apply(position)
        transformed_positions[k, :] = transformed
        
    nucleus_df['pt_position_x'] = transformed_positions[:, 0]
    nucleus_df['pt_position_y'] = transformed_positions[:, 1]
    nucleus_df['pt_position_z'] = transformed_positions[:, 2]
    
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
    if cells_df.empty:
        print("Warning: Empty dataframe provided to divide_volume_into_segments")
        return pd.DataFrame()
    
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
    if segments_df.empty:
        return segments_df
    
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
    if segments_df.empty:
        return segments_df
    
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

def add_layer_info(neurons_df, segments):
    """
    Add layer information to neurons based on their position
    
    Parameters:
        neurons_df: DataFrame with neuron information
        segments: DataFrame with layer segment information
    
    Returns:
        None: neurons_df is modified in-place
    """
    if neurons_df.empty or segments.empty:
        print("Warning: Empty dataframe provided to add_layer_info")
        return
    
    for (layer, ystart, yend) in segments[['layer', 'y_start', 'y_end']].values:
        mask = (neurons_df['pt_position_y'] >= ystart)  & (neurons_df['pt_position_y'] < yend)
        neurons_df.loc[mask, 'layer'] = layer
    return