import pytest
from io import StringIO

from _icet import Cluster
from icet.core.structure import Structure
from icet.core.lattice_site import LatticeSite

from ase.build import bulk
import numpy as np


def strip_surrounding_spaces(input_string: str) -> str:
    """
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    """
    s = []
    for line in StringIO(input_string):
        if len(line.strip()) == 0:
            continue
        s += [line.strip()]
    return '\n'.join(s)


@pytest.fixture
def lattice_sites(request):
    """Defines lattice sites used to initialize a cluster."""
    order = request.param
    lattice_sites = []
    if order == 'singlet':
        lattice_sites.append(LatticeSite(0, [1, 0, 0]))
    elif order == 'triplet':
        indices = [0, 1, 2]
        offsets = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]
        for ind, offset in zip(indices, offsets):
            lattice_site = LatticeSite(ind, offset)
            lattice_sites.append(lattice_site)
    return lattice_sites


@pytest.fixture
def structure():
    """Defines structure used to initialize a cluster."""
    prim = bulk('H', a=4.0, crystalstructure='sc').repeat((3, 1, 1))
    return Structure.from_atoms(prim)


@pytest.mark.parametrize('lattice_sites', [
    'singlet',
    'triplet'],
    indirect=['lattice_sites'])
def test_init(lattice_sites, structure):
    """Test initialization of Cluster object."""
    cluster = Cluster(lattice_sites, structure)
    assert isinstance(cluster, Cluster)
    assert len(cluster) == len(lattice_sites)
    assert cluster.order == len(lattice_sites)
    ret_lattice_sites = cluster.lattice_sites
    assert len(ret_lattice_sites) == len(lattice_sites)
    for ret_site, target_site in zip(ret_lattice_sites, lattice_sites):
        assert ret_site == target_site


@pytest.mark.parametrize('lattice_sites, target_radius', [
    ('singlet', 0),
    ('triplet', 8)],
    indirect=['lattice_sites'])
def test_radius(lattice_sites, target_radius, structure):
    """Test radius property."""
    cluster = Cluster(lattice_sites, structure)
    assert abs(cluster.radius - target_radius) < 1e-6


@pytest.mark.parametrize('lattice_sites, target_positions', [
    ('singlet', [[12, 0, 0]]),
    ('triplet', [[0, 0, 0], [16, 0, 0], [-4, 0, 0]])],
    indirect=['lattice_sites'])
def test_positions(lattice_sites, target_positions, structure):
    """Test positions property."""
    cluster = Cluster(lattice_sites, structure)
    ret_positions = cluster.positions
    assert len(ret_positions) == len(target_positions)
    for ret_pos, target_pos in zip(ret_positions, target_positions):
        assert np.allclose(ret_pos, target_pos)


@pytest.mark.parametrize('lattice_sites, target_distances', [
    ('singlet', []),
    ('triplet', [16, 4, 20])],
    indirect=['lattice_sites'])
def test_distances(lattice_sites, target_distances, structure):
    """Test get_distances function."""
    cluster = Cluster(lattice_sites, structure)
    assert np.allclose(cluster.distances, target_distances)


expected_singlet_string = """
================================== Cluster ==================================
 Order:      1
 Radius:     0
-----------------------------------------------------------------------------
 Unitcell index |   Unitcell offset   |    Position
-----------------------------------------------------------------------------
             0  |      1     0     0  |   12.000000    0.000000    0.000000
=============================================================================
"""

expected_triplet_string = """
================================== Cluster ==================================
 Order:      3
 Radius:     8
 Distances:  16  4  20
-----------------------------------------------------------------------------
 Unitcell index |   Unitcell offset   |    Position
-----------------------------------------------------------------------------
             0  |      0     0     0  |    0.000000    0.000000    0.000000
             1  |      1     0     0  |   16.000000    0.000000    0.000000
             2  |     -1     0     0  |   -4.000000    0.000000    0.000000
=============================================================================
"""


@pytest.mark.parametrize('lattice_sites, target_str', [
    ('singlet', expected_singlet_string),
    ('triplet', expected_triplet_string)],
    indirect=['lattice_sites'])
def test_string_representation(lattice_sites, target_str, structure):
    cluster = Cluster(lattice_sites, structure)
    assert strip_surrounding_spaces(cluster.__str__()) == strip_surrounding_spaces(target_str)
