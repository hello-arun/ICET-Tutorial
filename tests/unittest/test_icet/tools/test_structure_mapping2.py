import pytest
import numpy as np
from ase import Atom, Atoms
from ase.build import bulk, make_supercell
from icet import ClusterSpace
from icet.tools import map_structure_to_reference


testdata = [
    ([[4, 1, -3], [1, 3, 1], [-1, 1, 3]], {'Al': [0, 3, 9, 12], 'X': [1, 8, 13]},
     [1, 0.5, 0.8125, 0.4375, 2/3, 0.125, 5/8, 0.39583333, 0.6875, 5/8, 1/3, 5/8, 0.375]),
    ([[2, 0, 1], [0, 3, 1], [0, 1, 3]], {'Al': [0, 3, 9, 12], 'X': [1, 8, 13]},
     [1, 0.5, 0.8125, 0.375, 0.70833333, 0.20833333, 2/3, 5/12, 5/8, 5/8, 0, 5/8, 1/3]),
    ([[2, 0, 1], [0, 3, 1], [0, 1, 3]], {'Al': [3, 9], 'X': [2, 7, 8]},
     [1, 0.75, 0.8125, 0.59375, 5/8, 0.54166667, 31/48, 5/8, 5/8, 0.75, 0.5, 5/8, 7/16])]


@pytest.fixture
def ideal_structure() -> Atoms:
    # Construct reference structure and corresponding cluster space
    a = 5.0
    reference = bulk('Y', a=a, crystalstructure='fcc')
    reference.append(Atom('O', (1 * a / 4., 1 * a / 4., 1 * a / 4.)))
    reference.append(Atom('O', (3 * a / 4., 3 * a / 4., 3 * a / 4.)))
    return reference


@pytest.fixture
def cluster_space(ideal_structure) -> ClusterSpace:
    cs = ClusterSpace(ideal_structure, [5.0, 3.0], [['Y', 'Al'], ['O', 'X'], ['O', 'X']])
    return cs


@pytest.mark.parametrize("P,occupation,cv_target", testdata)
def test_mapping_with_occupation_only(ideal_structure, cluster_space, P, occupation, cv_target):
    structure = make_supercell(ideal_structure, P)
    for elem, occ in occupation.items():
        structure.symbols[occ] = elem
    cv = cluster_space.get_cluster_vector(structure)
    assert np.all(np.isclose(cv, cv_target))


@pytest.mark.parametrize('P,occupation,cv_target', testdata)
@pytest.mark.parametrize('strain', [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                    [[1.03, -0.01, 0], [0, 1, 0], [0, 0, 1]],
                                    [[1, -0.01, 0.01], [-0.01, 0.99, 0.01], [0, 0.01, 1.02]]])
@pytest.mark.parametrize('rattle', [0, 0.01])
def test_mapping(ideal_structure, cluster_space, P, occupation, cv_target, strain, rattle):
    structure = make_supercell(ideal_structure, P)
    for elem, occ in occupation.items():
        structure.symbols[occ] = elem

    # apply strain
    structure.set_cell(np.dot(structure.cell, strain), scale_atoms=True)

    # apply rattle
    structure.positions += rattle * (2 * np.random.random_sample((len(structure), 3)) - 1)

    # make sure the cluster vector calculation fails
    if np.isclose(np.linalg.det(strain), 1) and abs(rattle) > 0:
        with pytest.raises(RuntimeError):
            _ = cluster_space.get_cluster_vector(structure)
    elif not np.isclose(np.linalg.det(strain), 1):
        with pytest.raises(ValueError):
            _ = cluster_space.get_cluster_vector(structure)

    # map structure
    mapped_structure, _ = map_structure_to_reference(structure, ideal_structure,
                                                     inert_species=['Y', 'Al'])

    # check cluster vector
    cv = cluster_space.get_cluster_vector(mapped_structure)
    assert np.all(np.isclose(cv, cv_target))
