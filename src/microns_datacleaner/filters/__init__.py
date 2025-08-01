"""
# Filters subpackage

The filters subpackage helps to query the units and synapse tables. The tables are just Pandas Dataframes, so it is always possible to `.query` them.
However, often it is necessary to query for several aspects at once, which is inconvenient, especially for the synapses.
The `filters` package helps to reduce the effort for the most common operations. 
As stated in the Quick Start, the three most important functions are `filter_neurons`, `filter_connections` and `synapses_by_id`. 
There are several examples in the `basic_notebook`.  The API reference below contains detailed information about the arguments of these functions. 

> It is important to notice that the filters act only on the predefined columns of the unit table, but not on custom columns added from the other tables. In these cases, your best bet is to `.query` directly.

Please read the API below for more information in individual functions.
"""

from .filters import * 

__all__ = []