#!/usr/bin/env Python3

"""
This file contains unit tests and other tests. It can be executed by
simply executing this file from a shell prompt:

    $ ./test_variable_transformation.py

In which case it will use the system's default Python version. If a specific
Python version should be used, run that Python version with this file as input,
e.g.:

    python3 test_variable_transformation.py

For a description of the Python unit testing framework, see this link:
https://docs.python.org/3/library/unittest.html

When executing this file doc testing is also performed on all doc tests in
the cluster_space.py file

"""

import unittest
from io import StringIO

import numpy as np
from ase.build import fcc111
from icet import ClusterExpansion, ClusterSpace
from icet.core.lattice_site import LatticeSite
from icet.tools.variable_transformation import _is_site_group_in_orbit


def find_orbit_and_cluster_with_indices(orbit_list, site_indices):
    """
    Go through the orbit list and find the cluster with the specified list
    of site indices
    ----------
    orbit_list
        list of orbits
    site_indices
        list of lattice sites indices
    """

    for i in range(len(orbit_list)):
        orbit = orbit_list.get_orbit(i)

        # Check if the number of sites matches the order of the orbit
        if len(site_indices) != orbit.order:
            continue

        for cluster in orbit.clusters:

            # Check if the list of site indices matches those for the cluster
            if all(cluster.lattice_sites[j].index == site_indices[j]
                   for j in range(len(site_indices))):
                return orbit, cluster

        return None, None


def strip_surrounding_spaces(input_string):
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


def _assertEqualComplexList(self, retval, target):
    """
    Helper function that conducts a systematic comparison of a nested list
    with dictionaries.
    """
    self.assertIsInstance(retval, type(target))
    for row_retval, row_target in zip(retval, target):
        self.assertIsInstance(row_retval, type(row_target))
        for key, val in row_target.items():
            self.assertIn(key, row_retval)
            s = ['key: {}'.format(key)]
            s += ['type: {}'.format(type(key))]
            s += ['retval: {}'.format(row_retval[key])]
            s += ['target: {}'.format(val)]
            info = '   '.join(s)
            if isinstance(val, float):
                self.assertAlmostEqual(val, row_retval[key], places=9,
                                       msg=info)
            else:
                self.assertEqual(row_retval[key], val, msg=info)


unittest.TestCase.assertEqualComplexList = _assertEqualComplexList


def _assertAlmostEqualList(self, retval, target, places=6):
    """
    Helper function that conducts an element-wise comparison of two lists.
    """
    self.assertIsInstance(retval, type(target))
    self.assertEqual(len(retval), len(target))
    for k, (r, t) in enumerate(zip(retval, target)):
        s = ['element: {}'.format(k)]
        s += ['retval: {} ({})'.format(r, type(r))]
        s += ['target: {} ({})'.format(t, type(t))]
        info = '   '.join(s)
        self.assertAlmostEqual(r, t, places=places, msg=info)


unittest.TestCase.assertAlmostEqualList = _assertAlmostEqualList


class TestVariableTransformationTriplets(unittest.TestCase):
    """Container for test of the class functionality for a system with
    triplets."""

    def __init__(self, *args, **kwargs):
        super(TestVariableTransformationTriplets, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Au', 'Pd']
        self.cutoffs = [3.0, 3.0]
        structure_prim = fcc111('Au', a=4.0, size=(1, 1, 6), vacuum=10, periodic=True)
        structure_prim.wrap()
        self.structure_prim = structure_prim
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs, self.chemical_symbols)
        parameters = [0.0] * 4 + [0.1] * 6 + [-0.02] * 11
        self.ce = ClusterExpansion(self.cs, parameters)
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat((2, 2, 1))
        for i in range(len(self.supercell)):
            structure = self.supercell.copy()
            structure.symbols[i] = self.chemical_symbols[1]
            self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_is_site_group_in_orbit(self):
        """Tests _is_site_group_in_orbit functionality """
        # Check that a pair for which all sites have the different indices and but the same offset
        # gives true
        orbit, cluster = find_orbit_and_cluster_with_indices(self.cs.orbit_list, [2, 3])
        orbit.translate([2, 2, 2])
        self.assertTrue(_is_site_group_in_orbit(orbit, cluster.lattice_sites))

        # Check that a pair for which all sites have the different indices and one has been offset
        # gives false
        orbit, cluster = find_orbit_and_cluster_with_indices(self.cs.orbit_list, [2, 3])
        lattice_sites = cluster.lattice_sites
        lattice_sites[0] = LatticeSite(lattice_sites[0].index,
                                       lattice_sites[0].unitcell_offset + np.array([-2, -2, -2]))
        self.assertFalse(_is_site_group_in_orbit(orbit, lattice_sites))

        # Check that a triplet for which all sites have the same offset gives true
        orbit, cluster = find_orbit_and_cluster_with_indices(self.cs.orbit_list, [0, 0, 0])
        orbit.translate([2, 2, 2])
        self.assertTrue(_is_site_group_in_orbit(orbit, cluster.lattice_sites))

        # Check that a triplet in which one site has a different offset gives false
        orbit, cluster = find_orbit_and_cluster_with_indices(self.cs.orbit_list, [0, 0, 0])
        lattice_sites = cluster.lattice_sites
        lattice_sites[0] = LatticeSite(lattice_sites[0].index,
                                       lattice_sites[0].unitcell_offset + np.array([-2, -2, -2]))
        self.assertFalse(_is_site_group_in_orbit(orbit, lattice_sites))


if __name__ == '__main__':
    unittest.main()
