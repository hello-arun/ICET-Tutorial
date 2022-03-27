"""
This script checks that all atom objects in the database can have
its cluster vector computed
"""

import random
from icet import ClusterSpace


def generate_mixed_structure(structure_prim, chemical_symbols):
    """
    Generate a supercell structure based on the input structure and populate it
    randomly with the species specified.
    """
    repeat = [5, 5, 5]
    structure = structure_prim.copy().repeat(repeat)
    for at in structure:
        element = random.choice(chemical_symbols)
        at.symbol = element
    return structure


def generate_cluster_vector_set(n, structure_prim, chemical_symbols,
                                cluster_space):
    """
    Generate a set of cluster vectors from cluster space.
    """
    cluster_vectors = []
    for i in range(n):
        structure = generate_mixed_structure(structure_prim, chemical_symbols)
        cv = cluster_space.get_cluster_vector(structure)
        cluster_vectors.append(cv)

    return cluster_vectors


def test_cluster_vectors(structures_for_testing):
    chemical_symbols = ['H', 'He', 'Pb']
    for structure in structures_for_testing.values():
        cutoffs = [1.4] * 3
        cluster_space = ClusterSpace(structure, cutoffs, chemical_symbols)
        _ = generate_cluster_vector_set(3, structure, chemical_symbols, cluster_space)
