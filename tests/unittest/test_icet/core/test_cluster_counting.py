import pytest

from ase.build import bulk
from icet.core.orbit_list import OrbitList


@pytest.fixture
def orbit_list():
    """Create an orbit list based on a primitive hcp structure."""
    structure_prim = bulk('Ni', 'hcp', a=2.0)
    cutoffs = [2.2]
    chemical_symbols = [('Ni', 'Fe'), ('Ni', 'Fe')]
    symprec = 1e-5
    position_tolerance = 1e-5
    fractional_position_tolerance = 1e-6
    orbit_list = OrbitList(structure_prim, cutoffs, chemical_symbols,
                           symprec, position_tolerance,
                           fractional_position_tolerance)
    orbit_list.sort(position_tolerance)
    return orbit_list


@pytest.fixture
def structure():
    """Create an hcp supercell."""
    structure = bulk('Ni', 'hcp', a=2.0).repeat([2, 1, 1])
    structure.set_chemical_symbols('NiFeNi2')
    return structure


@pytest.mark.parametrize('orbit_indices, expected_counts', [
                         (None, {0: {('Fe',): 1, ('Ni',): 3},
                                 1: {('Fe', 'Fe'): 1, ('Fe', 'Ni'): 4, ('Ni', 'Ni'): 7},
                                 2: {('Fe', 'Ni'): 6, ('Ni', 'Ni'): 6}}),
                         ([2], {2: {('Fe', 'Ni'): 6, ('Ni', 'Ni'): 6}})])
def test_get_cluster_counts(orbit_list, structure, orbit_indices, expected_counts):
    """Tests the cluster counting functionality of the orbit list."""
    counts = orbit_list.get_cluster_counts(structure,
                                           fractional_position_tolerance=1e-6,
                                           orbit_indices=orbit_indices)
    assert len(counts) == len(expected_counts)
    for orbit_index in expected_counts:
        assert orbit_index in counts
        for species in expected_counts[orbit_index]:
            assert species in counts[orbit_index]
            assert expected_counts[orbit_index][species] == counts[orbit_index][species]
