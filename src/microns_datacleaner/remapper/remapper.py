import numpy as np
import pandas as pd
import scipy.sparse as sp

def get_id_map(units):
	"""
    Creates a mapping dictionary from original neuron IDs to sequential integer indices.
    This function generates a dictionary that maps the original neuron IDs to a sequential
	series of integers starting from 0.
    
    Parameters:
    -----------
        units: pandas.DataFrame
            The neuron properties DataFrame containing neuron metadata with a
            'pt_root_id' column that holds the original neuron identifiers.
    
    Returns:
    --------
        dict
            A dictionary mapping original neuron IDs (from 'pt_root_id' column)
            to sequential integer indices starting from 0. Keys are the original
            IDs and values are integers from 0 to len(units)-1.
    """

    N = len(units)

    # Declare new indices, and get the current ones
    idx_reset = {} 
    ids_original = units["pt_root_id"].values

    # Dictionary mapping the current indices to new ones
    for i in range(N):
        idx_reset[ids_original[i]] = i

    return idx_reset


def remap_table(idx_remap, table, columns):
	"""
    Remaps specified columns in a table using a provided ID mapping dictionary.
    This function applies an ID remapping transformation to specified columns
    of a DataFrame, converting original neuron IDs to sequential integer indices.
    
    Parameters:
    -----------
        idx_remap: dict
            Dictionary mapping original neuron IDs to new sequential integer IDs.
            Format: {original_id: new_sequential_id}. Can be obtained from get_id_map.
        table: pandas.DataFrame
            The DataFrame to be remapped. 
        columns: list
            List of column names that contain IDs to be remapped. These columns
            should contain values that exist as keys in the idx_remap dictionary.
    
    Returns:
    --------
        None.
            This function modifies the input table in-place and does not return
            a value. The specified columns are updated with the remapped IDs.
    """

    # Perform the remapping by mappling the dictionary to the 
    # corresponding columns
    table.loc[:, columns] = table[columns].map(idx_remap.get)


def remap_all_tables(units, connections):
	"""
    Remaps all neuron and connection tables to use sequential integer IDs.
    This convenience function performs a complete remapping of both neuron and
    connection tables from original ID systems to sequential integer indexing
    starting from 0.
    
    Parameters:
    -----------
        units: pandas.DataFrame
            The neuron properties DataFrame containing neuron metadata with
            original 'pt_root_id' identifiers.
        connections: pandas.DataFrame
            The synaptic connections DataFrame with 'pre_pt_root_id' and
            'post_pt_root_id' columns containing original neuron IDs.
    
    Returns:
    --------
        tuple of pandas.DataFrames
            new_units: DataFrame with both original 'pt_root_id' and remapped
            	'id_remapped' columns.
            new_connections: DataFrame with remapped 'pre_id' and 'post_id'
            	columns using sequential integer IDs.
    """
	
    # Duplicate the pt_root_id column, one will be remapped
    new_units = units.rename(columns={"pt_root_id" : "id_remapped"})
    new_units['pt_root_id'] = units['pt_root_id']

    # Prepare the new connections
    new_conns = connections.rename(columns={"pre_pt_root_id":"pre_id", "post_pt_root_id":"post_id"})

    # Get a dictionary matchking the new ids with the pt_root ones 
    idx_remap = get_id_map(units)

    # Remap the tables
    remap_table(idx_remap, new_conns, ["pre_id", "post_id"])
    remap_table(idx_remap, new_units, ["id_remapped"])

    return new_units, new_conns 
    #sp.csr_matrix(new_units)
