"""
Test structure numeration by checking that it yields the correct
number of structure.
"""

from ase.build import bulk, fcc100
from ase import Atom
from icet.tools import (enumerate_structures,
                        enumerate_supercells)
from icet import ClusterSpace
import numpy as np


def count_structures(structure, sizes, species, correct_count, tag,
                     conc_rest=None):
    """
    Count structures given by structure enumeration and assert that the
    right number is given.

    Parameters
    ----------
    structure : ASE Atoms
        Primitive structure for the enumeration.
    sizes : list of ints
        Cell sizes to be included in the enumeration.
    species : list of str
        Species to be passed to the enumeration.
    correct_count : int
        Expected number of structures.
    tag : str
        Describes the structure.
    """
    count = 0
    for _ in enumerate_structures(structure, sizes, species,
                                  concentration_restrictions=conc_rest):
        count += 1
    msg = 'Structure enumeration failed for {}'.format(tag)
    assert count == correct_count, msg


def count_and_check_unique_set(cluster_space, structure, sizes, species, correct_count,
                               tag, conc_rest=None):
    """
    Count structures given by structure enumeration and assert that the
    right number is given. Also make sure that structures produced
    all have unique cluster vectors, indicating that they are in fact
    inequivalent.

    Parameters
    ----------
    cluster_space : ClusterSpace
        Cluster space with which cluster vectors should be calculated.
    structure : ASE Atoms
        Primitive structure for the enumeration.
    sizes : list of ints
        Cell sizes to be included in the enumeration.
    species : list of str
        Species to be passed to the enumeration.
    correct_count : int
        Expected number of structures.
    tag : str
        Describes the structure.
    """
    count = 0
    cvs = []
    msg_equiv = 'Structure enumeration produced equivalent structures for {}'.format(tag)
    for structure in enumerate_structures(structure, sizes, species,
                                          concentration_restrictions=conc_rest):
        count += 1
        cv = cluster_space.get_cluster_vector(structure)
        for cv_comp in cvs:
            assert not np.allclose(cv_comp, cv), msg_equiv
        cvs.append(cv)
    msg = 'Structure enumeration failed for {}'.format(tag)
    assert count == correct_count, msg


def test_structure_enumeration():
    tag = 'FCC, 3 elements'
    structure = bulk('Au', crystalstructure='fcc')
    species = ['Au', 'Pd', 'Cu']
    sizes = range(1, 7)
    correct_count = 1081
    count_structures(structure, sizes, species, correct_count, tag)

    tag = 'FCC, elongated cell, two sites'
    structure = bulk('Au', crystalstructure='fcc', a=4.0)
    cell = structure.cell
    cell[0] = 1.33 * cell[0]
    structure.cell = cell
    structure.append(Atom('H', (2.0, 2.0, 2.0)))
    species = [['Au', 'Pd'], ['H', 'V']]
    sizes = range(1, 5)
    correct_count = 1500
    count_structures(structure, sizes, species, correct_count, tag)

    tag = 'HCP'
    structure = bulk('Au', crystalstructure='hcp', a=4.0)
    species = ['Au', 'Pd']
    sizes = range(1, 6)
    correct_count = 984
    count_structures(structure, sizes, species, correct_count, tag)

    tag = 'Surface'
    structure = fcc100('Au', (1, 1, 1), a=4.0, vacuum=2.0)
    species = ['Au', 'Pd']
    sizes = range(1, 9)
    correct_count = 271
    count_structures(structure, sizes, species, correct_count, tag)

    tag = 'Chain'
    structure = bulk('Au', a=4.0)
    structure.set_pbc((False, False, True))
    species = ['Au', 'Pd']
    sizes = range(1, 9)
    correct_count = 62
    count_structures(structure, sizes, species, correct_count, tag)

    tag = 'FCC, concentration restricted'
    structure = bulk('Au', crystalstructure='fcc')
    species = ['Au', 'Pd']
    sizes = range(1, 9)
    concentration_restrictions = {'Au': [0.0, 0.36]}
    correct_count = 134
    count_structures(structure, sizes, species, correct_count, tag,
                     conc_rest=concentration_restrictions)

    # Enumerate smaller sets but ensure that all structures produced are unique
    tag = 'FCC, small set'
    structure = bulk('Au', crystalstructure='fcc')
    species = ['Au', 'Pd']
    cluster_space = ClusterSpace(structure, [6.0, 5.0], species)
    sizes = range(1, 5)
    correct_count = 29
    count_and_check_unique_set(cluster_space, structure, sizes, species, correct_count, tag)

    tag = 'HCP, concentration_restricted 2/8'
    structure = bulk('Ti', crystalstructure='hcp', a=3.0)
    species = ['Ti', 'W']
    cluster_space = ClusterSpace(structure, [12.0], species)
    sizes = [4]
    concentration_restrictions = {'W': (2 / 8, 2 / 8)}
    correct_count = 35
    count_and_check_unique_set(cluster_space, structure, sizes, species, correct_count, tag,
                               conc_rest=concentration_restrictions)

    tag = 'HCP, concentration_restricted 4/8'
    structure = bulk('Ti', crystalstructure='hcp', a=3.0)
    species = ['Ti', 'W']
    cluster_space = ClusterSpace(structure, [8.0, 5.0, 5.0], species)
    sizes = [4]
    concentration_restrictions = {'W': (4 / 8, 4 / 8)}
    correct_count = 68
    count_and_check_unique_set(cluster_space, structure, sizes, species, correct_count, tag,
                               conc_rest=concentration_restrictions)

    # Enumerate supercells
    tag = 'FCC'
    structure = bulk('Au', crystalstructure='fcc')
    count = len(list(enumerate_supercells(structure, [6])))
    msg = 'Supercell enumeration failed for {}'.format(tag)
    assert count == 10, msg

    tag = 'FCC'
    structure = bulk('Au', crystalstructure='fcc')
    count = len(list(enumerate_supercells(structure, [6], niggli_reduce=False)))
    msg = 'Supercell enumeration failed for {}'.format(tag)
    assert count == 10, msg

    tag = 'FCC'
    structure = bulk('Au', crystalstructure='fcc')
    count = len(list(enumerate_supercells(structure, range(0, 6))))
    msg = 'Supercell enumeration failed for {}'.format(tag)
    assert count == 18, msg
