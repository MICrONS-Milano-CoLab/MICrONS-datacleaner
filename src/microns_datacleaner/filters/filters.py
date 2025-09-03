import pandas as pd
import numpy as np


def filter_neurons(units, layer=None, brain_area=None, cell_type=None, tuning=None,  proofread=None):
    """
    Filters neurons by several common characteristics simultaneously.
    This function allows filtering of neurons based on multiple criteria including
    layer, brain area, cell type, functional tuning properties, and proofreading
    status. Parameters left as None will not be used for filtering, allowing
    flexible combinations of filtering criteria.
    
    Parameters:
    -----------
        units: pandas.DataFrame
            The neuron properties DataFrame containing all neuron data to be filtered.
        layer: str or list, optional
            The layer(s) to filter for. Can be a single layer name as string or
            multiple layers as a list of strings, by default None.
        brain_area: str or list, optional
            The brain area(s) to filter for. Can be a single area name as string
            or multiple areas as a list of strings, by default None.
        cell_type: str or list, optional
            The cell type(s) to filter for. Can be a single cell type as string
            or multiple cell types as a list of strings, by default None.
        tuning: str, optional
            Functional tuning filter criteria. Use 'matched' to filter for
            functionally matched neurons, 'tuned' for neurons with tuning
            information, or 'untuned' for neurons without tuning, by default None.
        proofread: str, optional
            Proofreading level filter. Options include 'non' (no proofreading),
            'clean' (some proofreading), 'extended' (best proofreading). Note that
            'clean' includes both clean and extended neurons. Use prefixes 'ax_'
            or 'dn_' to target axons or dendrites respectively (e.g., 'dn_non',
            'ax_clean'), by default None.
    
    Returns:
    --------
        pandas.DataFrame
            A filtered DataFrame containing only the neurons that satisfy all
            specified filtering criteria.
    """

    # Initialize an empty query for pandas
    query = ""

    # Get the filters for layer if asked for them 
    # After each condition, an 'and' (&) is added to chain for the potential next one
    # Columns name do not need anything. variable names are preceded by @
    if type(layer) is str:
        query += "(layer == @layer)&" 
    elif type(layer) is list:
        query += "(layer in @layer)&" 

    # Same for cell types
    if type(cell_type) is str:
        query += "(cell_type == @cell_type)&"
    elif type(cell_type) is list:
        query += "(cell_type in @cell_type)&"

    # Same for brain area
    if type(brain_area) is str:
        query += "(brain_area == @brain_area)&"
    elif type(brain_area) is list:
        query += "(brain_area in @brain_area)&"

    # Get the filter for tuned/untuned neurons.
    if tuning == "matched":
        query += "(tuning_type != 'not_matched')&"
    elif not tuning is None:
        query += "(tuning_type == @tuning)&"

    # For proofread of axons and dendrites
    if proofread == 'ax_non':
        query += "(strategy_axon == 'none')&"
    elif proofread == 'ax_clean':
        query += "(strategy_axon != 'none')&"
    elif proofread == 'ax_extended':
        query += "(strategy_axon != 'axon_fully_extended')&"

    elif proofread == 'dn_non':
        query += "(strategy_dendrite == 'none')&"
    elif proofread == 'dn_clean':
        query += "(strategy_dendrite != 'none')&"
    elif proofread == 'dn_extended':
        query += "(strategy_dendrite == 'dendrite_extended')&"

    # The last character is always an '&' that needs to be removed. Then we query the table
    if len(query) > 0:
        return units.query(query[:-1])
    else:
        return units


def synapses_by_id(connections, pre_ids=None, post_ids=None, both=True):
    """
    This function extracts synapses from a connectivity DataFrame based on specified
    presynaptic and postsynaptic neuron IDs. The filtering behavior can be customized
    to return synapses that match either presynaptic OR postsynaptic criteria, or
    only those that match BOTH criteria simultaneously.
    
    Parameters:
    -----------
        connections: pandas.DataFrame
            The connectivity DataFrame containing synaptic connection information
            with columns for presynaptic and postsynaptic neuron IDs.
        pre_ids: array-like, optional
            Array of presynaptic neuron IDs to filter for. If None, presynaptic
            filtering is not applied, by default None.
        post_ids: array-like, optional
            Array of postsynaptic neuron IDs to filter for. If None, postsynaptic
            filtering is not applied, by default None.
        both: bool, optional
            Controls filtering behavior when both pre_ids and post_ids are specified.
            If True, returns only synapses where both presynaptic and postsynaptic
            IDs match the specified criteria (intersection). If False, returns
            synapses where either presynaptic OR postsynaptic IDs match (union),
            by default True.
    
    Returns:
    --------
        pandas.DataFrame
            A filtered DataFrame containing only the synapses that match the
            specified ID criteria.
    """

    if post_ids is None:
        return connections[connections["pre_pt_root_id"].isin(pre_ids)]
    if pre_ids is None:
        return connections[connections["post_pt_root_id"].isin(post_ids)]
    else:
        if both:
            return connections[connections["pre_pt_root_id"].isin(pre_ids) & connections["post_pt_root_id"].isin(post_ids)]
        else: 
            return connections[connections["pre_pt_root_id"].isin(pre_ids) | connections["post_pt_root_id"].isin(post_ids)]


def filter_connections(units, connections, layer=[None,None], tuning=[None,None], brain_area=[None, None], cell_type=[None,None], proofread=[None,None]):
    """
    Filters synaptic connections by applying separate criteria to presynaptic and postsynaptic neurons.
    This convenience function allows for independent filtering of presynaptic and postsynaptic neurons
    using different criteria, then returns all connections between the filtered neuron populations. 
    The function internally calls filter_neurons for each synaptic partner and synapses_by_id to extract
    matching connections.
    
    Parameters:
    -----------
        units: pandas.DataFrame
            The neuron properties DataFrame containing all neuron metadata.
        connections: pandas.DataFrame
            The synaptic connections DataFrame containing connectivity information.
        layer: list of length 2, optional
            Filtering criteria for cortical layers. First element applies to
            presynaptic neurons, second to postsynaptic neurons, by default [None, None].
        tuning: list of length 2, optional
            Filtering criteria for functional tuning properties. First element
            applies to presynaptic neurons, second to postsynaptic neurons, by default [None, None].
        brain_area : list of length 2, optional
            Filtering criteria for brain areas. First element applies to
            presynaptic neurons, second to postsynaptic neurons. Each element
            follows the same format as the brain_area parameter in filter_neurons
            (str or list of str), by default [None, None].
        cell_type: list of length 2, optional
            Filtering criteria for cell types. First element applies to
            presynaptic neurons, second to postsynaptic neurons, by default [None, None].
        proofread: list of length 2, optional
            Filtering criteria for proofreading levels. First element applies to
            presynaptic neurons, second to postsynaptic neurons,
            by default [None, None].
    
    Returns:
    --------
        pandas.DataFrame
            A filtered connections DataFrame containing only synapses between
            neurons that satisfy the specified presynaptic and postsynaptic
            filtering criteria.
    """
  
    neurons_pre = filter_neurons(units, layer=layer[0], tuning=tuning[0], cell_type=cell_type[0], brain_area=brain_area[0], proofread=proofread[0])
    neurons_post = filter_neurons(units, layer=layer[1], tuning=tuning[1], cell_type=cell_type[1], brain_area=brain_area[1], proofread=proofread[1])
    return synapses_by_id(connections, pre_ids=neurons_pre["pt_root_id"], post_ids=neurons_post["pt_root_id"], both=True)


def remove_autapses(connections):
    """
    Removes autapses from the synaptic connections table.
    
    This function filters out autaptic connections (synapses where a neuron
    connects to itself) from the provided connections DataFrame.
    
    Parameters:
    -----------
        connections: pandas.DataFrame
            The synaptic connections DataFrame containing connectivity information
            with columns for presynaptic and postsynaptic neuron IDs.
    
    Returns:
    --------
        pandas.DataFrame
            A new DataFrame containing all synaptic connections except autapses.
    """
   
    return connections[connections["pre_id"] != connections["post_id"]]


def connections_to(post_id, connections, only_id=True):
    """
    Retrieves presynaptic neurons that connect to a specified postsynaptic neuron.
    This function identifies all presynaptic neurons that form synaptic connections
    with the specified postsynaptic neuron ID.
    
    Parameters:
    -----------
        post_id: int or str
            The ID of the postsynaptic neuron for which to find presynaptic
            partners.
        connections: pandas.DataFrame
            The synaptic connections DataFrame containing connectivity information
            with columns for presynaptic and postsynaptic neuron IDs.
        only_id: bool, optional
            Controls the output format. If True, returns only the presynaptic
            neuron IDs. If False, returns the complete connection
            records as a DataFrame, by default True.
    
    Returns:
    --------
        pandas.DataFrame
            If only_id is True, returns the presynaptic neuron IDs. If only_id is False,
            returns a DataFrame with all connection records where the specified neuron is postsynaptic.
    """

    if only_id:
        return connections.loc[connections["post_id"] == post_id, "pre_id"]
    else:
        return connections[connections["post_id"] == post_id]


def connections_from(pre_id, connections, only_id=True):
    """
    Retrieves postsynaptic neurons that receive connections from a specified presynaptic neuron.
    This function identifies all postsynaptic neurons that receive synaptic connections
    from the specified presynaptic neuron ID.
    
    Parameters:
    -----------
        pre_id: int or str
            The ID of the presynaptic neuron for which to find postsynaptic
            targets.
        connections: pandas.DataFrame
            The synaptic connections DataFrame containing connectivity information
            with columns for presynaptic and postsynaptic neuron IDs.
        only_id: bool, optional
            Controls the output format. If True, returns only the postsynaptic
            neuron IDs. If False, returns the complete connection
            records as a DataFrame, by default True.
    
    Returns:
    --------
        pandas.DataFrame
            If only_id is True, returns the postsynaptic neuron IDs. If only_id is False,
            returns a DataFrame with all connection records where the specified neuron is presynaptic.
    """

    if only_id:
        return connections.loc[connections["pre_id"] == pre_id, "post_id"]
    else:
        return connections.loc[connections["pre_id"] == pre_id]

    
