"""
# Remapper subpackage

The remapper subpackage is particularly useful to work with functional data. The package distributes the average rates of every neuron in the functional
dataset to each direction (and orientation), i.e. the tuning curves. These can be used to estimate input current to neurons.

The most efficient way to perform this computation is to compute the adjacency matrix `msp` by the vector of rates `rates` for functionally matched neurons.  
However, most of the informatino is Pandas DataFrames, which complicate this operation. 

The remapper package helps to create the adjacency matrix of the data, by substituting all the `pt_root_id` identifiers by numbers from 
`0` to `N-1`, being `N` the number of selected functionally matched neurons. The remapped ids are sorted in the same order as the unit table
provided. In this way, if a row in the remapped table contains `i j s` the current input to `j` from `i` is `s * rates[i]`.

Assume `units_func` is a unit table with functionally matched neurons and `syn_func` contains synapses among them. Then,

```python
import scipy.sparse as sp
import microns_dataclear.remapper as rem

#Obtain the matrix of rates (one vector for each orientation)
rates = np.reshape(np.concatenate(units_func['rate_ori'].values), (-1, 8))

#Remap indices and build adjacency matrix
units_func, syn_func_remapped = rem.remap_all_tables(units_func, syn_func)
m = syn_func_remapped.values
msp = sp.csr_matrix((m[:,2], (m[:,0], m[:,1])), shape=(len(units_func), len(units_func))) 

#Compute input current to the neurons!
currents = msp @ rates
currents.sum(axis=0)
```

A fully developed example is present in the `basic_tutorial`.
"""
from .remapper import * 

__all__ = ["remap_all_tables"]