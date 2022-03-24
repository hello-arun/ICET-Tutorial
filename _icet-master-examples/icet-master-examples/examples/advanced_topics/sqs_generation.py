"""
This example demonstrates how to generate special quasirandom structure.
"""

# Import modules
from ase import Atom
from ase.build import bulk
from icet import ClusterSpace
from icet.tools.structure_generation import (generate_sqs,
                                             generate_sqs_from_supercells,
                                             generate_sqs_by_enumeration,
                                             generate_target_structure)

from icet.input_output.logging_tools import set_log_config
set_log_config(level='INFO')

# Generate SQS for binary fcc, 50 % concentration
primitive_structure = bulk('Au')
cs = ClusterSpace(primitive_structure, [8.0, 4.0], ['Au', 'Pd'])
target_concentrations = {'Au': 0.5, 'Pd': 0.5}
sqs = generate_sqs(cluster_space=cs,
                   max_size=8,
                   target_concentrations=target_concentrations)
print('Cluster vector of generated structure:', cs.get_cluster_vector(sqs))

# Generate SQS for binary fcc with specified supercells
supercells = [primitive_structure.repeat((1, 2, 4))]
sqs = generate_sqs_from_supercells(cluster_space=cs,
                                   supercells=supercells,
                                   n_steps=10000,
                                   target_concentrations=target_concentrations)
print('Cluster vector of generated structure:', cs.get_cluster_vector(sqs))

# Use enumeration to generate SQS for binary fcc, 50 % concentration
sqs = generate_sqs_by_enumeration(cluster_space=cs,
                                  max_size=8,
                                  target_concentrations=target_concentrations)
print('Cluster vector of generated structure:', cs.get_cluster_vector(sqs))

# Generate SQS for a system with two sublattices
primitive_structure = bulk('Au', a=4.0)
primitive_structure.append(Atom('H', position=(2.0, 2.0, 2.0)))
cs = ClusterSpace(primitive_structure, [7.0], [['Au', 'Cu', 'Pd'], ['H', 'X']])
print(cs)
# Target concentrations are specified per sublattice
target_concentrations = {'A': {'Au': 6 / 8, 'Cu': 1 / 8, 'Pd': 1 / 8},
                         'B': {'H': 1 / 4, 'X': 3 / 4}}
sqs = generate_sqs(cluster_space=cs,
                   max_size=16,
                   include_smaller_cells=False,
                   target_concentrations=target_concentrations,
                   n_steps=50000)
print('Cluster vector of generated structure:', cs.get_cluster_vector(sqs))

# Generate structure with a specified cluster vector
primitive_structure = bulk('Au')
cs = ClusterSpace(primitive_structure, [5.0], ['Au', 'Pd'])
target_cluster_vector = [1.0, 0.0] + [0.5] * (len(cs) - 2)
target_concentrations = {'Au': 0.5, 'Pd': 0.5}
sqs = generate_target_structure(cluster_space=cs,
                                max_size=8,
                                target_cluster_vector=target_cluster_vector,
                                target_concentrations=target_concentrations)
print('Cluster vector of generated structure:', cs.get_cluster_vector(sqs))
