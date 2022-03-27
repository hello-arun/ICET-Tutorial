#!/usr/bin/env Python3

"""
This file contains unit tests and other tests. It can be executed by
simply executing this file from a shell prompt:

    $ ./test_ground_state_finder.py

In which case it will use the system's default Python version. If a specific
Python version should be used, run that Python version with this file as input,
e.g.:

    python3 test_ground_state_finder.py

For a description of the Python unit testing framework, see this link:
https://docs.python.org/3/library/unittest.html

When executing this file doc testing is also performed on all doc tests in
the cluster_space.py file

"""

import pytest
import unittest
from unittest import mock
from io import StringIO
import inspect
import os
import sys
import numpy as np
import importlib

from ase import Atom
from ase.build import bulk, fcc111
from ase.db import connect as db_connect
from icet import ClusterExpansion, ClusterSpace
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
from icet.tools.variable_transformation import (transform_parameters, get_transformation_matrix)
try:
    import icet.tools.ground_state_finder
except ImportError as ex:
    module = ex.args[0].split()[0]
    if module == 'Python-MIP':
        raise unittest.SkipTest('no mip module')
    else:
        raise


def find_orbit_and_equivalent_site_with_indices(orbit_list, site_indices):
    """
    Go through the orbit list and find the equivalent with the specified list
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

        for sites in orbit.equivalent:

            # Check if the list of site indices matches those for the equivalent site
            if all(sites[j].index == site_indices[j] for j in range(len(site_indices))):
                return orbit, sites

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
                self.assertAlmostEqual(val, row_retval[key], places=9, msg=info)
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


class TestGroundStateFinder(unittest.TestCase):
    """Container for test of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinder, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Ag', 'Au']
        self.cutoffs = [4.3]
        self.structure_prim = bulk(self.chemical_symbols[1], a=4.0)
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs, self.chemical_symbols)
        self.ce = ClusterExpansion(self.cs, [0, 0, 0.1, -0.02])
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat(2)
        for i in range(len(self.supercell)):
            structure = self.supercell.copy()
            structure.symbols[i] = self.chemical_symbols[0]
            self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                                    verbose=False)

    def test_mip_import(self):
        """Tests the Python-MIP import statement"""
        # Test that an error is raised if Python-MIP is not installed
        with self.assertRaises(ImportError) as cm:
            with mock.patch.dict(sys.modules, {'mip': None}):
                importlib.reload(icet.tools.ground_state_finder)
        self.assertTrue('Python-MIP (https://python-mip.readthedocs.io/en/latest/) is required in '
                        'order to use the ground state finder.' in str(cm.exception))

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from GroundStateFinder instance
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        self.assertIsInstance(gsf, icet.tools.ground_state_finder.GroundStateFinder)

    def test_init_solver(self):
        """Tests that initialization of tested class work."""
        # initialize from GroundStateFinder instance
        # Set the solver explicitely
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               solver_name='CBC', verbose=False)
        self.assertEqual('CBC', gsf._model.solver_name.upper())

    def test_init_fails_for_ternary_with_one_active_sublattice(self):
        """Tests that initialization fails for a ternary system with one active
        sublattice."""
        chemical_symbols = ['Au', 'Ag', 'Pd']
        cs = ClusterSpace(self.structure_prim, cutoffs=self.cutoffs,
                          chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [0.0]*len(cs))
        with self.assertRaises(NotImplementedError) as cm:
            icet.tools.ground_state_finder.GroundStateFinder(ce, self.supercell, verbose=False)
        self.assertTrue('Currently, systems with more than two allowed species on any sublattice '
                        'are not supported.' in str(cm.exception))

    def test_optimization_status_property(self):
        """Tests the optimization_status property."""

        # Check that the optimization_status is None initially
        self.assertIsNone(self.gsf.optimization_status)

        # Check that the optimization_status is OPTIMAL if a ground state is found
        species_count = {self.chemical_symbols[0]: 1}
        self.gsf.get_ground_state(species_count=species_count, threads=1)
        self.assertEqual(str(self.gsf.optimization_status), 'OptimizationStatus.OPTIMAL')

    def test_model_property(self):
        """Tests the model property."""
        self.assertEqual(self.gsf.model.name, 'CE')

    @pytest.mark.coverage_hangup
    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure) for structure in self.all_possible_structures])

        # Provide counts for first species
        species_count = {self.chemical_symbols[0]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species0 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, target_val)

        # Provide counts for second species
        species_count = {self.chemical_symbols[1]: len(self.supercell) - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species1 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, predicted_species1)

        # Set the maximum run time
        species_count = {self.chemical_symbols[0]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, max_seconds=0.5,
                                                 threads=1)
        predicted_max_seconds = self.ce.predict(ground_state)
        self.assertGreaterEqual(predicted_max_seconds, predicted_species0)

    def test_get_ground_state_fails_for_faulty_species_to_count(self):
        """Tests that get_ground_state fails if species_to_count is faulty."""
        # Check that get_ground_state fails if counts are provided for multiple
        # species
        species_count = {self.chemical_symbols[0]: 1,
                         self.chemical_symbols[1]: len(self.supercell) - 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('Provide counts for at most one of the species on each active sublattice '
                        '({}), not {}!'.format(self.gsf._active_species,
                                               list(species_count.keys()))
                        in str(cm.exception))

        # Check that get_ground_state fails if counts are provided for a
        # species not found on the active sublattice
        species_count = {'H': 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The species {} is not present on any of the active sublattices'
                        ' ({})'.format(list(species_count.keys())[0], self.gsf._active_species)
                        in str(cm.exception))

        # Check that get_ground_state fails if the count exceeds the number sites on the active
        # sublattice
        faulty_species = self.chemical_symbols[0]
        faulty_count = len(self.supercell) + 1
        species_count = {faulty_species: faulty_count}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The count for species {} ({}) must be a positive integer and cannot '
                        'exceed the number of sites on the active sublattice '
                        '({})'.format(faulty_species, faulty_count, len(self.supercell))
                        in str(cm.exception))

        # Check that get_ground_state fails if the count is not a positive integer
        species = self.chemical_symbols[0]
        count = -1
        species_count = {species: count}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The count for species {} ({}) must be a positive integer and cannot '
                        'exceed the number of sites on the active sublattice '
                        '({})'.format(species, count, len(self.supercell)) in str(cm.exception))

    def test_create_cluster_maps(self):
        """Tests _create_cluster_maps functionality """
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        gsf._create_cluster_maps(self.structure_prim)

        # Test cluster to sites map
        target = [[0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.assertEqual(target, gsf._cluster_to_sites_map)

        # Test cluster to orbit map
        target = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        self.assertEqual(target, gsf._cluster_to_orbit_map)

        # Test ncluster per orbit map
        target = [1, 1, 6, 3]
        self.assertEqual(target, gsf._nclusters_per_orbit)


class TestGroundStateFinderInactiveSublattice(unittest.TestCase):
    """Container for test of the class functionality for a system with an
    inactive sublattice."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinderInactiveSublattice, self).__init__(*args, **kwargs)
        self.chemical_symbols = [['Ag', 'Au'], ['H']]
        self.cutoffs = [4.3]
        a = 4.0
        structure_prim = bulk(self.chemical_symbols[0][1], a=a)
        structure_prim.append(Atom(self.chemical_symbols[1][0], position=(a / 2, a / 2, a / 2)))
        self.structure_prim = structure_prim
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs, self.chemical_symbols)
        self.ce = ClusterExpansion(self.cs, [0, 0, 0.1, -0.02])
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat(2)
        sublattices = self.cs.get_sublattices(self.supercell)
        self.n_active_sites = [len(subl.indices) for subl in sublattices.active_sublattices]
        for i, sym in enumerate(self.supercell.get_chemical_symbols()):
            if sym not in self.chemical_symbols[0]:
                continue
            structure = self.supercell.copy()
            structure.symbols[i] = self.chemical_symbols[0][0]
            self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                                    verbose=False)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ClusterExpansion instance
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        self.assertIsInstance(gsf, icet.tools.ground_state_finder.GroundStateFinder)

    @pytest.mark.coverage_hangup
    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure)
                          for structure in self.all_possible_structures])

        # Provide counts for first species
        species_count = {self.chemical_symbols[0][0]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species0 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, target_val)

        # Provide counts for second species
        species_count = {self.chemical_symbols[0][1]: self.n_active_sites[0] - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species1 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, predicted_species1)

    def test_get_ground_state_fails_for_faulty_species_to_count(self):
        """Tests that get_ground_state fails if species_to_count is faulty."""
        # Check that get_ground_state fails if counts are provided for multiple species
        species_count = {self.chemical_symbols[0][0]: 1,
                         self.chemical_symbols[0][1]: self.n_active_sites[0] - 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('Provide counts for at most one of the species on each active sublattice '
                        '({}), not {}!'.format(self.gsf._active_species,
                                               list(species_count.keys()))
                        in str(cm.exception))

        # Check that get_ground_state fails if counts are provided for a
        # species not found on the active sublattice
        species_count = {'H': 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The species {} is not present on any of the active sublattices'
                        ' ({})'.format(list(species_count.keys())[0], self.gsf._active_species)
                        in str(cm.exception))

        # Check that get_ground_state fails if the count exceeds the number sites on the active
        # sublattice
        faulty_species = self.chemical_symbols[0][0]
        faulty_count = len(self.supercell)
        species_count = {faulty_species: faulty_count}
        n_active_sites = len([sym for sym in self.supercell.get_chemical_symbols() if
                              sym == self.chemical_symbols[0][1]])
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The count for species {} ({}) must be a positive integer and cannot '
                        'exceed the number of sites on the active sublattice '
                        '({})'.format(faulty_species, faulty_count, n_active_sites)
                        in str(cm.exception))

    def test_create_cluster_maps(self):
        """Tests _create_cluster_maps functionality """
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        gsf._create_cluster_maps(self.structure_prim)

        # Test cluster to sites map
        target = [[0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.assertEqual(target, gsf._cluster_to_sites_map)

        # Test cluster to orbit map
        target = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        self.assertEqual(target, gsf._cluster_to_orbit_map)

        # Test ncluster per orbit map
        target = [1, 1, 6, 3]
        self.assertEqual(target, gsf._nclusters_per_orbit)


class TestGroundStateFinderInactiveSublatticeSameSpecies(unittest.TestCase):
    """Container for test of the class functionality for a system with an
    inactive sublattice occupied by a species found on the active
    sublattice."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinderInactiveSublatticeSameSpecies, self).__init__(*args, **kwargs)
        self.chemical_symbols = [['Ag', 'Au'], ['Ag']]
        self.cutoffs = [4.3]
        a = 4.0
        structure_prim = bulk(self.chemical_symbols[0][1], a=a)
        structure_prim.append(Atom(self.chemical_symbols[1][0], position=(a / 2, a / 2, a / 2)))
        self.structure_prim = structure_prim
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs, self.chemical_symbols)
        self.ce = ClusterExpansion(self.cs, [0, 0, 0.1, -0.02])
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat(2)
        sublattices = self.cs.get_sublattices(self.supercell)
        self.n_active_sites = [len(subl.indices) for subl in sublattices.active_sublattices]
        for i, sym in enumerate(self.supercell.get_chemical_symbols()):
            if sym not in self.chemical_symbols[0]:
                continue
            structure = self.supercell.copy()
            structure.symbols[i] = self.chemical_symbols[0][0]
            self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                                    verbose=False)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ClusterExpansion instance
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        self.assertIsInstance(gsf, icet.tools.ground_state_finder.GroundStateFinder)

    @pytest.mark.coverage_hangup
    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure)
                          for structure in self.all_possible_structures])

        # Provide counts for first species
        species_count = {self.chemical_symbols[0][0]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species0 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, target_val)

        species_count = {self.chemical_symbols[0][1]: self.n_active_sites[0] - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species1 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, predicted_species1)

    def test_get_ground_state_fails_for_faulty_species_to_count(self):
        """Tests that get_ground_state fails if species_to_count is faulty."""
        # Check that get_ground_state fails if counts are provided for multiple species
        species_count = {self.chemical_symbols[0][0]: 1,
                         self.chemical_symbols[0][1]: self.n_active_sites[0] - 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('Provide counts for at most one of the species on each active sublattice '
                        '({}), not {}!'.format(self.gsf._active_species,
                                               list(species_count.keys()))
                        in str(cm.exception))

        # Check that get_ground_state fails if counts are provided for a
        # species not found on the active sublattice
        species_count = {'H': 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The species {} is not present on any of the active sublattices'
                        ' ({})'.format(list(species_count.keys())[0], self.gsf._active_species)
                        in str(cm.exception))

        # Check that get_ground_state fails if the count exceeds the number sites on the active
        # sublattice
        faulty_species = self.chemical_symbols[0][0]
        faulty_count = len(self.supercell)
        species_count = {faulty_species: faulty_count}
        n_active_sites = len([sym for sym in self.supercell.get_chemical_symbols() if
                              sym == self.chemical_symbols[0][1]])
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The count for species {} ({}) must be a positive integer and cannot '
                        'exceed the number of sites on the active sublattice '
                        '({})'.format(faulty_species, faulty_count, n_active_sites)
                        in str(cm.exception))

    def test_create_cluster_maps(self):
        """Tests _create_cluster_maps functionality """
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        gsf._create_cluster_maps(self.structure_prim)

        # Test cluster to sites map
        target = [[0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.assertEqual(target, gsf._cluster_to_sites_map)

        # Test cluster to orbit map
        target = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        self.assertEqual(target, gsf._cluster_to_orbit_map)

        # Test ncluster per orbit map
        target = [1, 1, 6, 3]
        self.assertEqual(target, gsf._nclusters_per_orbit)


class TestGroundStateFinderZeroParameter(unittest.TestCase):
    """Container for test of the class functionality for a system with a zero parameter."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinderZeroParameter, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Ag', 'Au']
        self.cutoffs = [4.3]
        self.structure_prim = bulk(self.chemical_symbols[1], a=4.0)
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs, self.chemical_symbols)
        nonzero_ce = ClusterExpansion(self.cs, [0, 0, 0.1, -0.02])
        lolg = LocalOrbitListGenerator(self.cs.orbit_list,
                                       Structure.from_atoms(self.structure_prim),
                                       self.cs.fractional_position_tolerance)
        full_orbit_list = lolg.generate_full_orbit_list()
        binary_parameters_zero = transform_parameters(self.structure_prim, full_orbit_list,
                                                      nonzero_ce.parameters)
        binary_parameters_zero[1] = 0
        A = get_transformation_matrix(self.structure_prim, full_orbit_list)
        Ainv = np.linalg.inv(A)
        zero_parameters = np.dot(Ainv, binary_parameters_zero)
        self.ce = ClusterExpansion(self.cs, zero_parameters)
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat(2)
        for i in range(len(self.supercell)):
            structure = self.supercell.copy()
            structure.symbols[i] = self.chemical_symbols[0]
            self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                                    verbose=False)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ClusterExpansion instance
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        self.assertIsInstance(gsf, icet.tools.ground_state_finder.GroundStateFinder)

    @pytest.mark.coverage_hangup
    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure) for structure in self.all_possible_structures])

        # Provide counts for first species
        species_count = {self.chemical_symbols[0]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species0 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, target_val)

        # Provide counts for second species
        species_count = {self.chemical_symbols[1]: len(self.supercell) - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species1 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, predicted_species1)


class TestGroundStateFinderTriplets(unittest.TestCase):
    """Container for test of the class functionality for a system with
    triplets."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinderTriplets, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Au', 'Pd']
        self.cutoffs = [3.0, 3.0]
        structure_prim = fcc111(self.chemical_symbols[0], a=4.0, size=(1, 1, 6), vacuum=10,
                                periodic=True)
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

    def setUp(self):
        """Setup before each test."""
        self.gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                                    verbose=False)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ClusterExpansion instance
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        self.assertIsInstance(gsf, icet.tools.ground_state_finder.GroundStateFinder)

    @pytest.mark.coverage_hangup
    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure) for structure in self.all_possible_structures])

        # Provide counts for first species
        species_count = {self.chemical_symbols[0]: len(self.supercell) - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species0 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, target_val)

        # Provide counts for second species
        species_count = {self.chemical_symbols[1]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species1 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, predicted_species1)

        # Check that get_ground_state finds 50-50 mix when no counts are provided
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        gs = gsf.get_ground_state(threads=1)
        self.assertEqual(gs.get_chemical_formula(), "Au12Pd12")

        # Ensure that an exception is raised when no solution is found
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               solver_name='CBC', verbose=False)
        species_count = {self.chemical_symbols[1]: 1}
        with self.assertRaises(Exception) as cm:
            gsf.get_ground_state(species_count=species_count, max_seconds=0.0, threads=1)
        self.assertTrue('Optimization failed' in str(cm.exception))


class TestGroundStateFinderTwoActiveSublattices(unittest.TestCase):
    """Container for test of the class functionality for a system with
    two active sublattices."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinderTwoActiveSublattices, self).__init__(*args, **kwargs)
        a = 4.0
        self.chemical_symbols = [['Au', 'Pd'], ['Li', 'Na']]
        self.cutoffs = [3.0]
        structure_prim = bulk(self.chemical_symbols[0][0], a=a)
        structure_prim.append(Atom(self.chemical_symbols[1][0], position=(a / 2, a / 2, a / 2)))
        structure_prim.wrap()
        self.structure_prim = structure_prim
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs,
                               self.chemical_symbols)
        parameters = [0.1, -0.45, 0.333, 2, -1.42, 0.98]
        self.ce = ClusterExpansion(self.cs, parameters)
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat(2)
        self.sl1_indices = [s for s, sym in enumerate(self.supercell.get_chemical_symbols()) if
                            sym == self.chemical_symbols[0][0]]
        self.sl2_indices = [s for s, sym in enumerate(self.supercell.get_chemical_symbols()) if
                            sym == self.chemical_symbols[1][0]]
        for i in self.sl1_indices:
            for j in self.sl2_indices:
                structure = self.supercell.copy()
                structure.symbols[i] = self.chemical_symbols[0][1]
                structure.symbols[j] = self.chemical_symbols[1][1]
                self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                                    verbose=False)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ClusterExpansion instance
        gsf = icet.tools.ground_state_finder.GroundStateFinder(self.ce, self.supercell,
                                                               verbose=False)
        self.assertIsInstance(gsf, icet.tools.ground_state_finder.GroundStateFinder)

    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure) for structure in self.all_possible_structures])

        # Provide counts for the first/first species
        species_count = {self.chemical_symbols[0][0]: len(self.sl1_indices) - 1,
                         self.chemical_symbols[1][0]: len(self.sl2_indices) - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count)
        predicted_species00 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species00, target_val)

        # Provide counts for the first/second species
        species_count = {self.chemical_symbols[0][0]: len(self.sl1_indices) - 1,
                         self.chemical_symbols[1][1]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count)
        predicted_species01 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species01, predicted_species00)

        # Provide counts for second/second species
        species_count = {self.chemical_symbols[0][1]: 1, self.chemical_symbols[1][1]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count)
        predicted_species11 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species11, predicted_species01)

    def _test_ground_state_cluster_vectors_in_database(self, db_name):
        """Tests get_ground_state functionality by comparing the cluster
        vectors for the structures in the databases."""

        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(filename))
        db = db_connect(os.path.join(path, db_name))

        # Select the structure set with the lowest pairwise correlations
        for index in [7, 13, 15, 62, 76]:
            row = db.get(f'id={index}')
            structure = row.toatoms()
            target_cluster_vector = self.cs.get_cluster_vector(structure)
            species_count = {
                self.chemical_symbols[0][0]:
                structure.get_chemical_symbols().count(self.chemical_symbols[0][0]),
                self.chemical_symbols[1][0]:
                structure.get_chemical_symbols().count(self.chemical_symbols[1][0])}
            ground_state = self.gsf.get_ground_state(species_count=species_count)
            gs_cluster_vector = self.cs.get_cluster_vector(ground_state)
            mean_diff = np.mean(np.abs(target_cluster_vector - gs_cluster_vector))
            self.assertLess(mean_diff, 1e-8)

    def test_ground_state_cluster_vectors(self):
        """Tests get_ground_state functionality by comparing the cluster
        vectors for ground states obtained from simulated annealing."""
        self._test_ground_state_cluster_vectors_in_database(
            '../../../structure_databases/annealing_ground_states.db')

    def test_get_ground_state_fails_for_faulty_species_to_count(self):
        """Tests that get_ground_state fails if species_to_count is faulty."""
        # Check that get_ground_state fails if counts are provided for a both species on one
        # of the active sublattices
        species_count = {self.chemical_symbols[0][0]: len(self.sl1_indices) - 1,
                         self.chemical_symbols[0][1]: 1, self.chemical_symbols[1][1]: 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('Provide counts for at most one of the species on each active sublattice '
                        '({}), not {}!'.format(self.gsf._active_species,
                                               list(species_count.keys()))
                        in str(cm.exception))

        # Check that get_ground_state fails if the count exceeds the number sites on the first
        # sublattice
        faulty_species = self.chemical_symbols[0][1]
        faulty_count = len(self.supercell)
        species_count = {faulty_species: faulty_count, self.chemical_symbols[1][1]: 1}
        n_active_sites = len([sym for sym in self.supercell.get_chemical_symbols() if
                              sym == self.chemical_symbols[0][0]])
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The count for species {} ({}) must be a positive integer and cannot '
                        'exceed the number of sites on the active sublattice '
                        '({})'.format(faulty_species, faulty_count, n_active_sites)
                        in str(cm.exception))

        # Check that get_ground_state fails if the count exceeds the number sites on the second
        # sublattice
        faulty_species = self.chemical_symbols[1][1]
        faulty_count = len(self.supercell)
        species_count = {faulty_species: faulty_count, self.chemical_symbols[0][1]: 1}
        n_active_sites = len([sym for sym in self.supercell.get_chemical_symbols() if
                              sym == self.chemical_symbols[1][0]])
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count)
        self.assertTrue('The count for species {} ({}) must be a positive integer and cannot '
                        'exceed the number of sites on the active sublattice '
                        '({})'.format(faulty_species, faulty_count, n_active_sites)
                        in str(cm.exception))

    def test_get_ground_state_passes_for_partial_species_to_count(self):
        # Check that get_ground_state passes if a single count is provided
        species_count = {self.chemical_symbols[0][1]: 1}
        self.gsf.get_ground_state(species_count=species_count)

        # Check that get_ground_state passes no counts are provided
        gs = self.gsf.get_ground_state()
        self.assertEqual(gs.get_chemical_formula(), "Au8Li8")


if __name__ == '__main__':
    unittest.main()
