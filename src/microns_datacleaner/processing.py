import numpy as np
import pandas as pd
import logging
from standard_transform import minnie_ds
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- MODULE CONSTANTS ---
LAYER_CELL_TYPES = {
    'L1': ['NGC', 'BPC', 'MC', 'BC'],
    'L2/3': ['23P'],
    'L4': ['4P'],
    'L5': ['5P-IT', '5P-ET', '5P-NP'],
    'L6': ['6P-IT', '6P-CT'],
    'WM': ['Oligo', 'OPC', 'Pericyte'],
}
"""Dictionary, whose keys are LAYER_ORDER. Each element includes a list with the cell types in each layer"""

LAYER_ORDER = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'WM']
"""Names of the layers."""

def merge_nucleus_with_cell_types(nucleus_df, cell_type_df):
    """
    Merges nucleus data with cell types
    Parameters:
    -----------
        nucleus_df: nucleus reference table
        cell_type_df: a classification table including cell types
    Returns:
    --------
        DataFrame merged with information about cell types
    """
    logging.info("Merging nucleus data with cell types.")
    if nucleus_df.empty or cell_type_df.empty:
        logging.error("Empty dataframe provided to merge_nucleus_with_cell_types.")
        raise ValueError('Empty dataframe provided to merge_nucleus_with_cell_types')

    #Perform a merge of both tables and keep only the desired columns
    merged = nucleus_df.merge(cell_type_df, left_on=['id'], right_on=['target_id'], how='inner')
    merged = merged[['id_x', 'pt_root_id_x', 'pt_position_x_x', 'pt_position_y_x', 'pt_position_z_x', 'classification_system', 'cell_type']]

    #Rename the columns since merge changes names, and return
    renamed_merged = merged.rename(
        columns={
            'id_x': 'id',
            'pt_root_id_x': 'pt_root_id',
            'pt_position_x_x': 'pt_position_x',
            'pt_position_y_x': 'pt_position_y',
            'pt_position_z_x': 'pt_position_z',
        }
    )
    logging.info("Successfully merged nucleus with cell types.")
    return renamed_merged

def merge_brain_area(nucleus_df, areas):
    """
    Merges nucleus data with brain area information
    Parameters:
    -----------
        nucleus_df: nucleus reference table
        areas: a classification table including brain areas 
    Returns:
    -------
        DataFrame merged with brain area information
    """
    logging.info("Merging nucleus data with brain area information.")
    if nucleus_df.empty or areas.empty:
        logging.error("Empty dataframe provided to merge_brain_area.")
        raise ValueError('Empty dataframe provided to merge_brain_area')

    #Perform a merge of both tables and keep only the desired columns
    merged = nucleus_df.merge(areas, left_on=['id'], right_on=['target_id'], how='inner')
    merged = merged[['id_x', 'pt_root_id_x', 'pt_position_x_x', 'pt_position_y_x', 'pt_position_z_x', 'classification_system', 'cell_type', 'tag']]

    #Rename the columns since merge changes names, and return
    renamed_merged = merged.rename(
        columns={
            'id_x': 'id',
            'pt_root_id_x': 'pt_root_id',
            'pt_position_x_x': 'pt_position_x',
            'pt_position_y_x': 'pt_position_y',
            'pt_position_z_x': 'pt_position_z',
            'tag': 'brain_area',
        }
    )
    logging.info("Successfully merged nucleus with brain area.")
    return renamed_merged

def merge_proofreading_status(nucleus_df, proofreading, version):
    """
    Merges nucleus data with proofreading status information
    Parameters:
    -----------
        nucleus_df: nucleus reference table
        proofreading: table including proofreading information 
    Returns:
    -------
        DataFrame merged with proofreading information
    """
    logging.info("Merging nucleus data with proofreading status.")
    if nucleus_df.empty:
        logging.error("Empty nucleus dataframe provided to merge_proofreading_status.")
        raise ValueError('Empty nucleus dataframe provided to merge_proofreading_status')

    #Perform a merge of both tables and keep only the desired columns
    merged = nucleus_df.merge(proofreading, left_on=['pt_root_id'], right_on=['pt_root_id'], how='left')

    name_axon = "strategy_axon" if version > 700 else "status_axon"
    name_dend = "strategy_dendrite" if version > 700 else "status_dendrite"
    logging.info(f"Using column names: {name_dend} and {name_axon} for version {version}.")

    merged = merged[
        [
            'pt_root_id', 'id_x', 'pt_position_x_x', 'pt_position_y_x', 'pt_position_z_x',
            'classification_system', 'cell_type', 'brain_area', name_dend, name_axon
        ]
    ]

    # Tag the ones that have not been proofread
    merged.loc[merged[name_dend].isna(), name_dend] = 'none'
    merged.loc[merged[name_axon].isna(), name_axon] = 'none'
    logging.info("Filled missing proofreading statuses with 'none'.")

    renamed_merged = merged.rename(columns={'id_x': 'id', 'pt_position_x_x': 'pt_position_x', 'pt_position_y_x': 'pt_position_y', 'pt_position_z_x': 'pt_position_z'})
    logging.info("Successfully merged with proofreading status.")
    return renamed_merged

def merge_functional_properties(nucleus_df, functional, best_only=True):
    """
    Merges nucleus data with functional properties
    Parameters:
    ------------
        nucleus_df: DataFrame with nucleus information
        functional: DataFrame with functional properties
        use_directions: Whether to use directions for angle calculations (default: False)
    Returns:
    --------
        DataFrame merged with functional properties
    """
    logging.info("Merging nucleus data with functional properties.")
    if nucleus_df.empty or functional.empty:
        logging.warning('Empty dataframe provided to merge_functional_properties')
        return nucleus_df

    if best_only:
        logging.info("Using only the best functional properties based on 'cc_abs'.")
        functional = functional.sort_values(by='cc_abs', ascending=False)
        functional = functional.drop_duplicates(subset='target_id', keep='first')
        functional = functional[['target_id', 'pt_root_id', 'pref_ori', 'pref_dir', 'gOSI', 'gDSI', 'cc_abs']]
    else:
        logging.info("Using all functional properties.")
        functional = functional[['target_id', 'pt_root_id', 'session', 'scan_idx', 'unit_id', 'pref_ori', 'pref_dir', 'gOSI', 'gDSI', 'cc_abs']]
        functional = functional.rename(columns={'unit_id': 'functional_unit_id'})

    #Needed just to filter potential pt root = 0, then we can drop
    functional = functional[functional['pt_root_id'] != 0]
    functional = functional.drop(columns=['pt_root_id'])

    # Then proceed on the merge. In this case there is no common columns so filtering and renaming the result
    # after the merge is not necessary anymore
    merged = nucleus_df.merge(functional, left_on=['id'], right_on=['target_id'], how='left')
    final_merged = merged.drop(columns=['target_id'])
    logging.info("Successfully merged with functional properties.")
    return final_merged

def transform_positions(df, x_col='pt_position_x', y_col='pt_position_y', z_col='pt_position_z'):
    """
    Transforms positions from voxels to μm
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with position columns to transform
    x_col : str, optional
        Name of the x-coordinate column (default: 'pt_position_x')
    y_col : str, optional
        Name of the y-coordinate column (default: 'pt_position_y')
    z_col : str, optional
        Name of the z-coordinate column (default: 'pt_position_z')
    Returns:
    --------
    pandas.DataFrame
        DataFrame with transformed positions
    """
    logging.info("Transforming positions from voxels to micrometers.")
    if df.empty:
        logging.warning("Empty dataframe provided to transform_positions")
        return df

    # Check if required columns exist
    if not all(col in df.columns for col in [x_col, y_col, z_col]):
        logging.error(f"Required columns {x_col}, {y_col}, {z_col} not found in the dataframe.")
        raise ValueError(f"Required columns {x_col}, {y_col}, {z_col} not found in the dataframe")

    transformed_positions = np.empty((len(df), 3))
    for k, (x, y, z) in enumerate(tqdm(df[[x_col, y_col, z_col]].values, desc="Transform positions")):
        position = np.array([x, y, z])
        transformed = minnie_ds.transform_vx.apply(position)
        transformed_positions[k, :] = transformed

    df[x_col] = transformed_positions[:, 0]
    df[y_col] = transformed_positions[:, 1]
    df[z_col] = transformed_positions[:, 2]
    logging.info("Position transformation complete.")
    return df

def divide_volume_into_segments(cells_df, segment_size=10.0):
    """
    Divide the volume into segments along the y-axis.
    Parameters:
    -----------
        cells_df: DataFrame with cell information
        segment_size (optioanl) Size of each segment in micrometers. Defaults to 10.0
    Returns:
    --------
        A DataFrame with the segmented layer information
    """
    logging.info(f"Dividing volume into segments of size {segment_size} micrometers.")
    if cells_df.empty:
        logging.warning("Empty dataframe provided to divide_volume_into_segments")
        raise ValueError('Warning: Empty dataframe provided to divide_volume_into_segments')

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

        segments.append(
            {
                'y_start': y_start,
                'y_end': y_end,
                'y_center': y_center,
                'L1_cells': layer_counts.get('L1', 0),
                'L2/3_cells': layer_counts.get('L2/3', 0),
                'L4_cells': layer_counts.get('L4', 0),
                'L5_cells': layer_counts.get('L5', 0),
                'L6_cells': layer_counts.get('L6', 0),
                'WM_cells': layer_counts.get('WM', 0),
                'dominant_layer': dominant_layer,
            }
        )

    segments_df = pd.DataFrame(segments)
    logging.info("Volume segmentation complete.")
    return segments_df



def merge_segments_by_layer(segments_df):
    """
    Merge segments that belong to the same layer.
    Parameters:
    -----	
        segments_df: DataFrame of segments to be merged
    Returns:
    --------
        DataFrame with merged layer segments
    """
    logging.info("Merging segments by layer.")
    if segments_df.empty:
        logging.warning("Empty dataframe provided to merge_segments_by_layer.")
        return segments_df

    layer_groups = {}
    current_layer = None
    start_idx = 0

    for i, row in segments_df.iterrows():
        layer = row['dominant_layer']
        if layer == 'Unknown' or layer == 'WM':
            continue
        if current_layer != layer:
            if current_layer is not None:
                if current_layer not in layer_groups:
                    layer_groups[current_layer] = []
                layer_groups[current_layer].append({'y_start': segments_df.iloc[start_idx]['y_start'], 'y_end': segments_df.iloc[i - 1]['y_end']})
            current_layer = layer
            start_idx = i

    if current_layer is not None and start_idx < len(segments_df):
        if current_layer not in layer_groups:
            layer_groups[current_layer] = []
        layer_groups[current_layer].append({'y_start': segments_df.iloc[start_idx]['y_start'], 'y_end': segments_df.iloc[len(segments_df) - 1]['y_end']})

    merged_layers = []
    for layer, regions in layer_groups.items():
        for i, region in enumerate(regions):
            merged_layers.append({
                'layer': layer, 'region_index': i,
                'y_start': region['y_start'], 'y_end': region['y_end'],
                'height': region['y_end'] - region['y_start'],
            })

    merged_df = pd.DataFrame(merged_layers)
    logging.info("Segments merged by layer.")
    return merged_df.sort_values('y_start')

def add_layer_info(neurons_df, segments):
    """
    Add layer information to neurons based on their position
    Parameters:
    -----------
        neurons_df: DataFrame with neuron information. Modified in-place.
        segments: DataFrame with layer segment information
    Returns:
    --------
        None 
    """
    logging.info("Adding layer information to neurons.")
    if neurons_df.empty or segments.empty:
        logging.warning("Empty dataframe provided to add_layer_info")
        return

    for layer, ystart, yend in segments[['layer', 'y_start', 'y_end']].values:
        mask = (neurons_df['pt_position_y'] >= ystart) & (neurons_df['pt_position_y'] < yend)
        neurons_df.loc[mask, 'layer'] = layer
    return