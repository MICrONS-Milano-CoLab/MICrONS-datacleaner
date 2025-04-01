from caveclient import CAVEclient
from standard_transform import minnie_ds 

import numpy as np
import pandas as pd
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
    return merged.rename(columns = {'id_x' : 'id', 'pt_root_id_x' : 'pt_rood_id', 'pt_position_x_x' : 'pt_position_x','pt_position_y_x' : 'pt_position_y','pt_position_z_x' : 'pt_position_z'  }) 
    
    
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