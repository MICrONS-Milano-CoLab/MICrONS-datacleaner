import pandas as pd
import numpy as np

class MicronsFilters:
    """
    Allows to easily query unit and synapse table
    """ 

    def __init__(self):
        pass

    def filter_neurons(self, units, layer=None, brain_area=None, cell_type=None, tuning=None,  proofread=None):
        """
        Filter neurons by several common characteristics at the same time.
        Leave parameters to None to not filter for them (default). Returns the table of the 
        neurons fulfilling the criteria.

        Parameters:
            units: DataFrame
                neuron's properties DataFrame
            layer : list or string 
                The layer(s) we want to filter for. One can filter for multiple layers by providing a list of strings with the 
                desired layers.
            cell_type : list or string
                The cell type(s) we want to filter for. Works like 'layer'. 
            brain_area : list or string
                The brain area(s) we want to filter for. Works like 'layer'. 
            tuning : string  
                If functionally matched neurons are present, these are tagged as "matched". If tuning information is available in the table,
                it is possible to filter also for "tuned" and "untuned" neurons.  
            proofread : list or string 
                Filters for different levels of proofreading, labelled as "non" (no proofreading),  "clean" (some kind of proofreading), 
                "extended" (best possible proofreading). Observe that `clean` includes both clean and extended.  
                Use a prefix `ax_` or `dn_` in order to target axons or dendrites, respectively. 
                Caution! no differences are made between all classes of clean axons.
                Valid examples are `dn_non` or `ax_clean`. 
        """

        #Initialize an empty query for pandas
        query = ""

        #Get the filters for layer if asked for them 
        #After each condition, an 'and' (&) is added to chain for the potential next one
        #Columns name do not need anything. variable names are preceded by @
        if type(layer) is str:
            query += "(layer == @layer)&" 
        elif type(layer) is list:
            query += "(layer in @layer)&" 

        #Same for cell types
        if type(cell_type) is str:
            query += "(cell_type == @cell_type)&"
        elif type(cell_type) is list:
            query += "(cell_type in @cell_type)&"

        #Same for brain area
        if type(brain_area) is str:
            query += "(brain_area == @brain_area)&"
        elif type(brain_area) is list:
            query += "brain_area in @brain_area)&"

        #Get the filter for tuned/untuned neurons.
        if tuning == "matched":
            query += "(tuning_type != 'not_matched')&"
        else:
            query += "(tuning_type) == @tuning"

        #For proofread of axons and dendrites
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

        #The last character is always an '&' that needs to be removed. Then we query the table
        return units.query(query[:-1])


    def synapses_by_id(self, connections, pre_ids=None, post_ids=None, both=True):
        """
        Given the ids of the neurons we want to filter for, grab the synapses that have ids matching for
        the pre- or post- synaptic neurons (or both).

        Parameters

        connections : DataFrame
            Dataframe with connectivity information
        pre_ids, post_ids : array
            These two arguments indicate the neuron ids that will be selected. If one of these is set to None, only pre- or post- synapses are considered.
            If both are specified, the behaviour can depend on 'both' parameter
        both : boolean 
            If True (default), only the inner set is returned, i.e., only synapses having both the pre and post indicated are returned. If False, any synapse
            having one pre OR post id is returned.
        """

        if post_ids is None:
            return connections[connections["pre_id"].isin(pre_ids)]
        if pre_ids is None:
            return connections[connections["post_id"].isin(post_ids)]
        else:
            if both:
                return connections[connections["pre_id"].isin(pre_ids) & connections["post_id"].isin(post_ids)]
            else: 
                return connections[connections["pre_id"].isin(pre_ids) | connections["post_id"].isin(post_ids)]


    def filter_connections(self, units, connections, layer=[None,None], tuning=[None,None], brain_area=[None, None], cell_type=[None,None], proofread=[None,None]):
        """
        Convenience function to separately filter pre- and postsynaptic neurons by a criterium and 
        then returning all connections fulfilling these conditions. 
        Internally it calls filter_neurons and synapses_by_id.

        Parameters
        ===========

        units : Dataframe   
            The units table
        connections: Dataframe   
            The synapse table
        layer : 2-element list
            Each element is a condition for the layer. First one affects presynaptic neurons, second affects postsynaptic ones.
            It is formatted the same way than the layer selection in `filter_neurons`. 
        tuning : 2-element list
            Similar to layer, following  the format for tuning selection in `filter_neurons`. 
        cell_type : 2-element list
            Similar to layer, following  the format for cell type selection in `filter_neurons`. 
        brain_area: 2-element list
            Similar to layer, following  the format for brain area selection in `filter_neurons`. 
        proofread : 2-element list
            Similar to layer, following  the format for proofread selection in `filter_neurons`. 

        Returns
        ========

            A new synapse table with the selected neurons 
        """

        neurons_pre = self.filter_neurons(units, layer=layer[0], tuning=tuning[0], cell_type=cell_type[0], brain_area=brain_area[0], proofread=proofread[0])
        neurons_post = self.filter_neurons(units, layer=layer[1], tuning=tuning[1], cell_type=cell_type[1], brain_area=brain_area[1], proofread=proofread[1])
        return self.synapses_by_id(connections, pre_ids=neurons_pre["id"], post_ids=neurons_post["id"], both=True)


    def remove_autapses(self, connections):
        """
        Returns a new copy of provided synapse table without autapses.
        """
        return connections[connections["pre_id"] != connections["post_id"]]

    def connections_to(self, post_id, connections, only_id=True):
        """
        Get the presynaptic neurons pointing to post_id. If only_id = True (the default) only the ids are returned, instead of the full table.
        """

        if only_id:
            return connections.loc[connections["post_id"] == post_id, "pre_id"]
        else:
            return connections[connections["post_id"] == post_id]

    def connections_from(self, pre_id, connections, only_id=True):
        """
        Get the postsynaptic neurons to which pre_id points. If only_id = True (the default) only the ids are returned, instead of the full table.  
        """
        if only_id:
            return connections.loc[connections["pre_id"] == pre_id, "post_id"]
        else:
            return connections.loc[connections["pre_id"] == pre_id]

        