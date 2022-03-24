"""
This example demonstrates how to construct cluster vectors.
"""

# Import modules
from ase.build import bulk
from icet import ClusterSpace

# Create a primitive structure, decide which additional elements to populate
# it with (Si, Ge) and set the cutoffs for pairs (5.0 Å), triplets (5.0 Å)
# and quadruplets (5.0 Å).
primitive_structure = bulk('Si')
cutoffs = [5.0, 5.0, 5.0]
subelements = ['Si', 'Ge']

# Initiate and print the cluster space.
cluster_space = ClusterSpace(primitive_structure, cutoffs, subelements)
print(cluster_space)

# Generate and print the cluster vector for a pure Si 2x2x2 supercell.
structure_1 = bulk('Si').repeat(2)
cluster_vector_1 = cluster_space.get_cluster_vector(structure_1)
print(cluster_vector_1)

# Generate and print the cluster vector for a mixed Si-Ge 2x2x2 supercell
structure_2 = bulk('Si').repeat(2)
structure_2[0].symbol = 'Ge'
cluster_vector_2 = cluster_space.get_cluster_vector(structure_2)
print(cluster_vector_2)
