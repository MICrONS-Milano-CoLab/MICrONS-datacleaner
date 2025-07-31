import numpy as np
import pandas as pd
import scipy.sparse as sp

def get_id_map(units):
    """
    Returns a dictionary that allows one to change the system's ids to
    to integer numbers starting at 0

    Parameters
    ===========

        units : dataframe
            The unit table
    """

    N = len(units)

    #Declare new indices, and get the current ones
    idx_reset = {} 
    ids_original = units["pt_root_id"].values

    #Dictionary mapping the current indices to new ones
    for i in range(N):
        idx_reset[ids_original[i]] = i

    return idx_reset
    #"""

def remap_table(idx_remap, table, columns):
    """
    Remap the table to have all IDs using unique integer numbers from
    0 to the number of neurons minus one. [0, N-1]. 

    Parameters:
    ==========
    idx_remap : dict 
        a dictionary in the format idx_remap[pt_root_id] = new_id. Can be obtained
        from get_id_map.
    table : DataFrame T
        The table to be remapped  
    columns : list 
        A list with the names of the columns that need to be remaped
    """

    #Perform the remapping by mappling the dictionary to the 
    #corresponding columns
    table.loc[:, columns] = table[columns].map(idx_remap.get)

def remap_all_tables(units, connections):
    """
    Perform the remap of all the three tables: neurons, connections and activity.
    This is needed for further processing. 

    Parameters
    ==========

        unit : dataframe
            The functional unit table 
        connections: dataframe
            Table of synapses between the units 
    """

    #Duplicate the pt_root_id column, one will be remapped
    new_units = units.rename(columns={"pt_root_id" : "id_remapped"})
    new_units['pt_root_id'] = units['pt_root_id']

    #Prepare the new connections
    new_conns = connections.rename(columns={"pre_pt_root_id":"pre_id", "post_pt_root_id":"post_id"})

    #Get a dictionary matchking the new ids with the pt_root ones 
    idx_remap = get_id_map(units)

    #Remap the tables
    remap_table(idx_remap, new_conns, ["pre_id", "post_id"])
    remap_table(idx_remap, new_units, ["id_remapped"])

    return new_units, new_conns 
    #sp.csr_matrix(new_units)