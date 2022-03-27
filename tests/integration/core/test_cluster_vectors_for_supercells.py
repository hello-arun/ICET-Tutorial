"""
Tests the calculation of singlet terms in a binary supercell.
"""

import itertools
import numpy.testing as npt

from ase.build import bulk, cut
from icet import ClusterSpace


def test_cluster_vectors_for_supercells():

    # initializes cluster space and get the internal primitive structure
    prim = bulk('Au', a=4.0, crystalstructure='hcp')
    subelements = ['Au', 'Pd']
    cutoffs = [0.0]

    cs = ClusterSpace(prim, cutoffs, subelements)
    structure_prim = cs.primitive_structure

    # create a supercell using permutation matrix
    p_trial = [[1, 0, 0], [0, 1, 5], [0, 0, 2]]
    supercell = cut(structure_prim, p_trial[0], p_trial[1], p_trial[2])

    # setup cartesian input to generate a random population
    cartesian_product_input = []
    for i in range(len(supercell)):
        cartesian_product_input.append(['Pd', 'Au'])

    # loop over element combinations and assert expected singlet value
    for subset in itertools.product(*cartesian_product_input):
        for atom, element in zip(supercell, subset):
            atom.symbol = element
        cv = cs.get_cluster_vector(supercell)
        expected_singlet = -supercell.get_chemical_symbols().count('Pd')
        expected_singlet += supercell.get_chemical_symbols().count('Au')
        expected_singlet /= len(supercell)
        npt.assert_almost_equal(cv[1], expected_singlet)
