#!/usr/bin/env Python3

"""
This file contains unit tests and other tests. It can be executed by
simply executing this file from a shell prompt:

    $ ./test_cluster_space.py

In which case it will use the system's default Python version. If a specific
Python version should be used, run that Python version with this file as input,
e.g.:

    python3 test_cluster_space.py

For a description of the Python unit testing framework, see this link:
https://docs.python.org/3/library/unittest.html

When executing this file doc testing is also performed on all doc tests in
the cluster_space.py file

"""

from collections import OrderedDict
from io import StringIO
import inspect
import os
import sys
import tempfile
import unittest

import numpy as np
from ase.build import bulk
from ase.db import connect as db_connect
from ase import Atoms  # NOQA (needed for eval(retval))
from icet import ClusterSpace


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


def _assertEqualAtoms(self, retval, target, places=6):
    """
    Helper function that conducts a basic comparison of two Atoms objects.
    """
    self.assertIsInstance(retval, type(target))
    self.assertEqual(len(retval), len(target))
    self.assertTrue(all(retval.pbc == target.pbc))
    self.assertAlmostEqualList(retval.cell.flatten(), target.cell.flatten(), places=places)
    for atom1, atom2 in zip(retval, target):
        self.assertEqual(atom1.symbol, atom2.symbol)
        self.assertAlmostEqualList(atom1.position, atom2.position, places=places)


unittest.TestCase.assertEqualAtoms = _assertEqualAtoms


class TestClusterSpace(unittest.TestCase):
    """Container for test of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterSpace, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Ag', 'Au']
        self.cutoffs = [4.0] * 3
        self.primitive_structure = bulk('Ag', a=4.09)
        self.structure_list = []
        for k in range(4):
            structure = self.primitive_structure.repeat(2)
            symbols = [self.chemical_symbols[0]] * len(structure)
            symbols[:k] = [self.chemical_symbols[1]] * k
            structure.set_chemical_symbols(symbols)
            self.structure_list.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.cs = ClusterSpace(self.primitive_structure, self.cutoffs,
                               self.chemical_symbols)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ASE Atoms
        cs = ClusterSpace(self.primitive_structure, self.cutoffs, self.chemical_symbols)
        self.assertIsInstance(cs, ClusterSpace)
        self.assertEqual(len(cs), len(self.cs))

    def test_init_fails_for_non_pbc(self):
        """Tests that initialization fails if pbc is false."""
        structure_surface = self.primitive_structure.copy()
        structure_surface.pbc = [1, 1, 0]
        with self.assertRaises(ValueError) as cm:
            ClusterSpace(structure_surface, self.cutoffs, self.chemical_symbols)
        self.assertTrue('Input structure must have periodic boundary '
                        'condition' in str(cm.exception))

    def test_init_fails_for_faulty_chemical_symbols(self):
        """Tests that initialization fails if chemical_symbols is faulty."""
        structure = bulk('Ag', a=4.09).repeat(2)
        chemical_symbols1 = ['Ag', 'Pd']
        chemical_symbols2 = [['Ag', 'Pd']] * len(structure)
        chemical_symbols3 = [('Ag', 'Pd')] * len(structure)

        # no problems
        ClusterSpace(structure, self.cutoffs, chemical_symbols1)
        ClusterSpace(structure, self.cutoffs, chemical_symbols2)
        ClusterSpace(structure, self.cutoffs, chemical_symbols3)

        # bad type
        chemical_symbols_bad = chemical_symbols2 + ['Ag']
        with self.assertRaises(TypeError) as cm:
            ClusterSpace(structure, self.cutoffs, chemical_symbols_bad)
        self.assertIn(
            'chemical_symbols must be List[str] or List[List[str]]', str(cm.exception))

        # bad length
        chemical_symbols_bad = chemical_symbols2 + [['Ag', 'Pd']]
        with self.assertRaises(ValueError) as cm:
            ClusterSpace(structure, self.cutoffs, chemical_symbols_bad)
        self.assertIn(
            'chemical_symbols must have same length as structure', str(cm.exception))

        # duplicate symbols for site
        chemical_symbols_bad = [['Ag', 'Pd']] * len(structure)
        chemical_symbols_bad[0] = ['Ag', 'Pd', 'Pd']
        with self.assertRaises(ValueError) as cm:
            ClusterSpace(structure, self.cutoffs, chemical_symbols_bad)
        self.assertIn('Found duplicates of allowed chemical symbols', str(cm.exception))

        # no active sites
        chemical_symbols_bad = [['Ag']] * 4 + [['Pd']] * 4
        with self.assertRaises(ValueError) as cm:
            ClusterSpace(structure, self.cutoffs, chemical_symbols_bad)
        self.assertIn('No active sites found', str(cm.exception))

    def test_len(self):
        """Tests length functionality."""
        number_orbits = self.cs.__len__()
        self.assertEqual(number_orbits, len(self.cs.orbit_list) + 1)

    def test_symprec(self):
        """Tests symprec property."""
        for val in [1e-3, 0.02]:
            cs = ClusterSpace(self.primitive_structure, self.cutoffs,
                              self.chemical_symbols, symprec=val)
            self.assertEqual(cs.symprec, val)
        with self.assertRaises(AttributeError) as cm:
            cs.symprec = 0.01
        self.assertIn("can't set attribute", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            cs = ClusterSpace(self.primitive_structure, self.cutoffs,
                              self.chemical_symbols, symprec=0)
        self.assertIn('must be a positive number', str(cm.exception))

    def test_position_tolerance(self):
        """Tests position_tolerance property."""
        for val in [1e-3, 0.02]:
            cs = ClusterSpace(self.primitive_structure, self.cutoffs,
                              self.chemical_symbols, position_tolerance=val)
            self.assertEqual(cs.position_tolerance, val)
        for val in [3e-3, 0.05]:
            cs = ClusterSpace(self.primitive_structure, self.cutoffs,
                              self.chemical_symbols, symprec=val)
            self.assertEqual(cs.position_tolerance, cs.symprec)
        with self.assertRaises(AttributeError) as cm:
            cs.position_tolerance = 0.01
        self.assertIn("can't set attribute", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            cs = ClusterSpace(self.primitive_structure, self.cutoffs,
                              self.chemical_symbols, position_tolerance=0)
        self.assertIn('must be a positive number', str(cm.exception))

    def test_space_group(self):
        """Tests space_group property."""
        cs = ClusterSpace(bulk('Au', crystalstructure='fcc', a=2.0), [1.9], ['Au', 'Ag'])
        self.assertEqual(cs.space_group, 'Fm-3m (225)')
        cs = ClusterSpace(bulk('W', crystalstructure='bcc', a=2.0), [1.9], ['Au', 'Ag'])
        self.assertEqual(cs.space_group, 'Im-3m (229)')
        cs = ClusterSpace(bulk('Mg', crystalstructure='hcp', a=2.0, c=3.2), [1.9], ['Au', 'Ag'])
        self.assertEqual(cs.space_group, 'P6_3/mmc (194)')
        cs = ClusterSpace(bulk('C', crystalstructure='diamond', a=2.0), [1.9], ['Au', 'Ag'])
        self.assertEqual(cs.space_group, 'Fd-3m (227)')
        cs = ClusterSpace(bulk('ZnS', crystalstructure='zincblende', a=2.0),
                          [1.9], [['Au', 'Ag'], ['Cl']])
        self.assertEqual(cs.space_group, 'F-43m (216)')

    def test_orbit_data(self):
        """Tests orbit_data property."""
        target = [OrderedDict([('index', 0),
                               ('order', 0),
                               ('radius', 0),
                               ('multiplicity', 1),
                               ('orbit_index', -1)]),
                  OrderedDict([('index', 1), ('order', 1),
                               ('radius', 0.0),
                               ('multiplicity', 1),
                               ('orbit_index', 0),
                               ('multicomponent_vector', [0])]),
                  OrderedDict([('index', 2), ('order', 2),
                               ('radius', 1.4460333675264896),
                               ('multiplicity', 6),
                               ('orbit_index', 1),
                               ('multicomponent_vector', [0, 0])]),
                  OrderedDict([('index', 3), ('order', 3),
                               ('radius', 1.6697355079971996),
                               ('multiplicity', 8),
                               ('orbit_index', 2),
                               ('multicomponent_vector', [0, 0, 0])]),
                  OrderedDict([('index', 4), ('order', 4),
                               ('radius', 1.771021950739177),
                               ('multiplicity', 2),
                               ('orbit_index', 3),
                               ('multicomponent_vector', [0, 0, 0, 0])])]
        self.assertEqualComplexList(self.cs.orbit_data, target)

    def test_repr(self):
        """Tests __repr__ method."""
        retval = self.cs.__repr__()
        self.assertIn('ClusterSpace', retval)
        self.assertIn('Atoms', retval)
        self.assertIn('cutoffs=', retval)
        self.assertIn('chemical_symbols=', retval)
        ret = eval(retval)
        self.assertIsInstance(ret, ClusterSpace)

    def test_str(self):
        """Tests __str__ method."""
        retval = self.cs.__str__()
        target = """
====================================== Cluster Space ======================================
 space group                            : Fm-3m (225)
 chemical species                       : ['Ag', 'Au'] (sublattice A)
 cutoffs                                : 4.0000 4.0000 4.0000
 total number of parameters             : 5
 number of parameters by order          : 0= 1  1= 1  2= 1  3= 1  4= 1
 fractional_position_tolerance          : 2e-06
 position_tolerance                     : 1e-05
 symprec                                : 1e-05
-------------------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multicomponent_vector | sublattices
-------------------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .           |      .
   1  |   1   |   0.0000 |        1     |       0     |          [0]          |      A
   2  |   2   |   1.4460 |        6     |       1     |        [0, 0]         |     A-A
   3  |   3   |   1.6697 |        8     |       2     |       [0, 0, 0]       |    A-A-A
   4  |   4   |   1.7710 |        2     |       3     |     [0, 0, 0, 0]      |   A-A-A-A
===========================================================================================
"""
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_get_string_representation(self):
        """Tests _get_string_representation functionality."""
        retval = self.cs._get_string_representation(print_threshold=2, print_minimum=1)
        target = """
====================================== Cluster Space ======================================
 space group                            : Fm-3m (225)
 chemical species                       : ['Ag', 'Au'] (sublattice A)
 cutoffs                                : 4.0000 4.0000 4.0000
 total number of parameters             : 5
 number of parameters by order          : 0= 1  1= 1  2= 1  3= 1  4= 1
 fractional_position_tolerance          : 2e-06
 position_tolerance                     : 1e-05
 symprec                                : 1e-05
-------------------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multicomponent_vector | sublattices
-------------------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .           |      .
 ...
   4  |   4   |   1.7710 |        2     |       3     |     [0, 0, 0, 0]      |   A-A-A-A
===========================================================================================
"""
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_print_overview(self):
        """Tests print_overview functionality."""
        with StringIO() as capturedOutput:
            sys.stdout = capturedOutput  # redirect stdout
            self.cs.print_overview()
            sys.stdout = sys.__stdout__  # reset redirect
            self.assertTrue('Cluster Space' in capturedOutput.getvalue())

    def test_get_number_of_orbits_by_order(self):
        """Tests get_number_of_orbits_by_order functionality """
        retval = self.cs.get_number_of_orbits_by_order()
        target = OrderedDict([(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
        self.assertEqual(target, retval)

    def test_get_cluster_vector(self):
        """Tests get_cluster_vector functionality."""
        target_cluster_vectors = [
            [1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, -0.75, 0.5, -0.25, 0.0],
            [1.0, -0.5, 0.16666666666666666, 0.0, 0.0],
            [1.0, -0.25, 0.0, 0.0, 0.0]]
        s = ['Error in test setup;']
        s += ['number of cluster vectors ({})'.format(
            len(target_cluster_vectors))]
        s += ['does not match']
        s += ['number of structures ({})'.format(len(self.structure_list))]
        info = ' '.join(s)
        self.assertEqual(len(target_cluster_vectors), len(self.structure_list),
                         msg=info)
        for structure, target in zip(self.structure_list, target_cluster_vectors):
            retval = list(self.cs.get_cluster_vector(structure))
            self.assertAlmostEqualList(retval, target, places=9)

        # Bad position
        structure = self.primitive_structure.repeat(3)
        structure[0].position[0] += 0.1
        with self.assertRaises(RuntimeError) as cm:
            self.cs.get_cluster_vector(structure)
        self.assertIn('Failed to find site by position', str(cm.exception))

    def test_get_coordinates_of_representative_cluster(self):
        """ Tests get_coordinates_of_representative_cluster functionality """
        cs = ClusterSpace(self.primitive_structure, self.cutoffs, self.chemical_symbols)

        # check calculation of positions for all the orbit indices for the Cluster Space
        for orbit_index in range(len(cs.orbit_list)):
            # check number of positions for each orbit index
            self.assertEqual(len(cs.get_coordinates_of_representative_cluster(orbit_index)),
                             orbit_index+1)

        # check positions of each atom for largest value of orbit_index
        positions_for_last_orbit_index = [[0., -2.045, -2.045], [-2.045,  0., -2.045],
                                          [-2.045, -2.045,  0.], [0., 0., 0.]]

        for i in range(len(positions_for_last_orbit_index)):
            self.assertAlmostEqualList(cs.get_coordinates_of_representative_cluster(3)[i].tolist(),
                                       positions_for_last_orbit_index[i])

        # check raised exception if orbit_index outside of the list of possible values
        with self.assertRaises(ValueError) as cm:
            cs.get_coordinates_of_representative_cluster(4)
        self.assertTrue('The input orbit index is not in the list of possible values.'
                        in str(cm.exception))

    def test_cutoffs(self):
        """Tests cutoffs property."""
        self.assertEqual(self.cs.cutoffs, self.cutoffs)

    def _test_cluster_vectors_in_database(self, db_name):
        """Tests the cluster vectors in the database."""

        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(filename))
        db = db_connect(os.path.join(path, db_name))

        entry1 = db.get(id=1)
        structure = entry1.toatoms()
        elements = entry1.data.elements
        cutoffs = entry1.data.cutoffs
        cs = ClusterSpace(structure, cutoffs, elements)

        for row in db.select():
            structure = row.toatoms()
            retval = cs.get_cluster_vector(structure)
            target = np.array(row.data.target_cv)
            self.assertTrue(np.all(np.isclose(target, retval)))

    def test_cluster_vectors_fcc(self):
        """
        Tests the calculation of cluster vectors against databases
        of structures with known cluster vectors.
        """
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/fcc_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/fcc_skew_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/fcc_ternary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/fcc_quaternary.db')

    def test_cluster_vectors_bcc(self):
        """
        Tests the calculation of cluster vectors against databases
        of structures with known cluster vectors.
        """
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/bcc_longedge_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/bcc_ternary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/bcc_quaternary.db')

    def test_cluster_vectors_hcp(self):
        """
        Tests the calculation of cluster vectors against databases
        of structures with known cluster vectors.
        """
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/hcp_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/hcp_skew_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/hcp_ternary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/hcp_quaternary.db')

        self._test_cluster_vectors_in_database(
            '../../../structure_databases/tetragonal_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/tetragonal_ternary.db')

    def test_read_write(self):
        """Tests read/write functionality."""
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()
        self.cs.write(f.name)

        cs_read = ClusterSpace.read(f.name)
        self.assertEqualAtoms(self.cs._input_structure, cs_read._input_structure)
        self.assertEqual(list(self.cs._cutoffs), list(cs_read._cutoffs))
        self.assertEqual(self.cs._input_chemical_symbols,
                         cs_read._input_chemical_symbols)
        os.remove(f.name)

    def test_chemical_symbols(self):
        """Tests chemical_symbols property."""
        target = [['Ag', 'Au']]
        self.assertEqual(self.cs.chemical_symbols, target)

    def test_prune_orbit_list(self):
        """Tests pruning internal orbit list."""
        orig_size = len(self.cs.orbit_list)
        prune_indices = [0, 1, 3, 2]
        self.cs._prune_orbit_list(indices=prune_indices)
        self.assertEqual(orig_size - len(prune_indices), len(self.cs.orbit_list))

    def test_copy(self):
        """ Tests copy function. """
        cs_copy = self.cs.copy()
        self.assertEqual(str(cs_copy), str(self.cs))

    def test_assert_structure_compatibility(self):
        """ Tests assert_structure_compatibility functionality """
        supercell = self.primitive_structure.repeat((2, 3, 4))

        # real supercell works
        self.cs.assert_structure_compatibility(supercell)

        # faulty volume
        supercell_tmp = supercell.copy()
        supercell_tmp.set_cell(1.01 * supercell_tmp.cell, scale_atoms=True)
        with self.assertRaises(ValueError) as cm:
            self.cs.assert_structure_compatibility(supercell_tmp)
        self.assertIn('Volume per atom of structure', str(cm.exception))
        self.assertIn('does not match', str(cm.exception))

        # faulty occupations
        supercell_tmp = supercell.copy()
        symbols = supercell_tmp.get_chemical_symbols()
        symbols[0] = 'W'
        supercell_tmp.set_chemical_symbols(symbols)
        with self.assertRaises(ValueError) as cm:
            self.cs.assert_structure_compatibility(supercell_tmp)
        self.assertIn('Occupations of structure not compatible', str(cm.exception))

        # PBC False
        structure = self.primitive_structure.copy()
        structure.pbc = False
        with self.assertRaises(ValueError) as cm:
            self.cs.assert_structure_compatibility(structure)
        self.assertIn('must have periodic boundary conditions', str(cm.exception))

    def test_get_possible_orbit_occupations(self):
        """Tests get possible orbit occupations."""

        occupation_singlet = self.cs.get_possible_orbit_occupations(
            orbit_index=0)
        self.assertEqual(occupation_singlet, [('Ag',), ('Au',)])

        occupation_pair = self.cs.get_possible_orbit_occupations(orbit_index=1)
        self.assertEqual(occupation_pair, [
                         ('Ag', 'Ag'), ('Ag', 'Au'), ('Au', 'Ag'), ('Au', 'Au')])

        occupation_triplet = self.cs.get_possible_orbit_occupations(
            orbit_index=2)
        self.assertEqual(occupation_triplet, [('Ag', 'Ag', 'Ag'),
                                              ('Ag', 'Ag', 'Au'),
                                              ('Ag', 'Au', 'Ag'),
                                              ('Ag', 'Au', 'Au'),
                                              ('Au', 'Ag', 'Ag'),
                                              ('Au', 'Ag', 'Au'),
                                              ('Au', 'Au', 'Ag'),
                                              ('Au', 'Au', 'Au')])

    def test_is_supercell_self_interacting(self):
        """ Test is_supercell_self_interacting function. """
        prim = bulk('Al', a=4.0)
        cs = ClusterSpace(prim, [4.1], ['Al', 'H'])
        supercell1 = prim.repeat(2)
        supercell2 = prim.repeat(3)
        self.assertTrue(cs.is_supercell_self_interacting(supercell1))
        self.assertFalse(cs.is_supercell_self_interacting(supercell2))

        # test sensitivity to symprec
        alat = 1.0
        clat = alat*1.63
        rcut = alat * 1.01
        symprec = 0.09
        fcc = bulk('Au', a=alat, cubic=True)
        hcp = bulk('Au', crystalstructure='hcp', a=alat, c=clat)
        for prim in [fcc, hcp]:
            cs = ClusterSpace(prim, [rcut], ['Au', 'Ag'], symprec=symprec)
            for n in range(2, 12):
                structure = prim.repeat(n)
                if n <= 2:
                    self.assertTrue(cs.is_supercell_self_interacting(structure))
                else:
                    self.assertFalse(cs.is_supercell_self_interacting(structure))


class TestClusterSpaceTernary(unittest.TestCase):
    """
    Container for tests of the class functionality for non-periodic structures
    """

    def __init__(self, *args, **kwargs):
        super(TestClusterSpaceTernary, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Ag', 'Au', 'Pd']
        self.cutoffs = [4.0] * 3
        self.primitive_structure = bulk('Ag', 'fcc')

    def setUp(self):
        """Instantiates class before each test."""
        self.cs = ClusterSpace(self.primitive_structure, self.cutoffs, self.chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def _get_multicomponent_vector(self, cluster_space, orbit_index):
        """
        Helper function to  return the mc vectors for a
        particular orbit.

        Parameters
        ----------
        cluster_space : icet cluster space
        orbit_index : int
            The orbit which the mc vectors should be returned from.
        """
        orbit = cluster_space.orbit_list.get_orbit(orbit_index)
        mc_vectors = [el['multicomponent_vector'] for el in orbit.cluster_vector_elements]
        return mc_vectors

    def _get_multicomponent_vector_permutations(self, cluster_space, orbit_index):
        """
        Helper function to  return the mc vectors for a
        particular orbit.

        Parameters
        ----------
        cluster_space : icet cluster space
        orbit_index : int
            The orbit which the permutations should be returned from.
        """
        orbit = cluster_space.orbit_list.get_orbit(orbit_index)
        permutations = [el['site_permutations'] for el in orbit.cluster_vector_elements]
        return permutations

    def test_multicomponent_cluster_vector_permutation(self):
        """Tests the multicomponent permutation functionality."""
        # Test orbit number 1
        orbit_index = 1
        mc_vector_target = [[0, 0], [0, 1], [1, 1]]
        mc_vector_retval = self._get_multicomponent_vector(self.cs, orbit_index)
        self.assertEqual(mc_vector_retval, mc_vector_target)

        permutations_target = [[[0, 1]],
                               [[0, 1], [1, 0]],
                               [[0, 1]]]
        permutation_retval = self._get_multicomponent_vector_permutations(self.cs, orbit_index)
        self.assertEqual(permutations_target, permutation_retval)

        # Test orbit number 2
        orbit_index = 2
        mc_vector_target = [[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [1, 1, 1]]
        mc_vector_retval = self._get_multicomponent_vector(self.cs, orbit_index)
        self.assertEqual(mc_vector_retval, mc_vector_target)

        permutations_target = [[[0, 1, 2]],
                               [[0, 1, 2], [0, 2, 1], [2, 0, 1]],
                               [[0, 1, 2], [1, 0, 2], [1, 2, 0]],
                               [[0, 1, 2]]]
        permutation_retval = self._get_multicomponent_vector_permutations(self.cs, orbit_index)
        self.assertEqual(permutations_target, permutation_retval)

        # Test orbit 3
        orbit_index = 3
        mc_vector_target = [[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1]]
        mc_vector_retval = self._get_multicomponent_vector(self.cs, orbit_index)
        self.assertEqual(mc_vector_retval, mc_vector_target)

        permutations_target = [[[0, 1, 2, 3]],
                               [[0, 1, 2, 3], [0, 1, 3, 2], [0, 3, 1, 2], [3, 0, 1, 2]],
                               [[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 3, 1], [2, 0, 1, 3],
                                [2, 0, 3, 1], [2, 3, 0, 1]],
                               [[0, 1, 2, 3], [1, 0, 2, 3], [1, 2, 0, 3], [1, 2, 3, 0]],
                               [[0, 1, 2, 3]]]
        permutation_retval = self._get_multicomponent_vector_permutations(self.cs, orbit_index)
        self.assertEqual(permutations_target, permutation_retval)


class TestClusterSpaceMultiSublattice(unittest.TestCase):
    """Container for test of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterSpaceMultiSublattice, self).__init__(*args, **kwargs)
        self.chemical_symbols = [['Ag', 'Au'],
                                 ['H', 'V']]
        self.cutoffs = [5] * 2
        self.primitive_structure = bulk('Ag', a=4.09, crystalstructure='bcc', cubic=True)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.cs = ClusterSpace(self.primitive_structure, self.cutoffs,
                               self.chemical_symbols)
        self.cluster_space_binary = ClusterSpace(
            self.primitive_structure, self.cutoffs, ['Ag', 'Au'])

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ASE Atoms
        cs = ClusterSpace(self.primitive_structure, self.cutoffs, self.chemical_symbols)
        self.assertIsInstance(cs, ClusterSpace)
        self.assertEqual(len(cs), len(self.cs))

    def test_correct_number_of_singlets(self):
        """Tests that we get two singlets."""
        singlet_count = 0
        for orbit in self.cs.orbit_list.orbits:
            if orbit.order == 1:
                singlet_count += 1
        self.assertEqual(singlet_count, 2)

    def test_correct_number_of_pairs(self):
        """Tests that we get correct number of pairs."""
        pair_counts = OrderedDict()
        pair_counts_binary = OrderedDict()
        for orbit in self.cs.orbit_list.orbits:
            if orbit.order == 2:
                radius = np.round(orbit.representative_cluster.radius, 3)
                if radius in pair_counts.keys():
                    pair_counts[radius] += 1
                else:
                    pair_counts[radius] = 1

        for orbit in self.cluster_space_binary.orbit_list.orbits:
            if orbit.order == 2:
                radius = np.round(orbit.representative_cluster.radius, 3)
                if radius in pair_counts_binary.keys():
                    pair_counts_binary[radius] += 1
                else:
                    pair_counts_binary[radius] = 1

        self.assertEqual(len(pair_counts.keys()),
                         len(pair_counts_binary.keys()))

        # origin to center atom only one since these are only
        # sublattice 1 -> sublattice 2 interactions
        self.assertEqual(pair_counts_binary[1.771], pair_counts[1.771])

        # Twice as many pairs in 100 direction since they can be both
        # sublattice-1 -> sublatice 1 and sublattice 2  -> sublattice 2
        self.assertEqual(pair_counts_binary[2.045] * 2, pair_counts[2.045])


class TestClusterSpaceMergedOrbits(unittest.TestCase):
    """
    Container for tests of the class functionality for merged orbits
    """

    def __init__(self, *args, **kwargs):
        super(TestClusterSpaceMergedOrbits, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Au', 'Pd']
        self.cutoffs = [5.0, 5.0]
        self.primitive_structure = bulk('Au', a=4.0)
        self.structure = self.primitive_structure.repeat((3, 3, 3))
        for i in [3, 7, 9, 14, 18, 21]:
            self.structure[i].symbol = 'Pd'

    def setUp(self):
        """Instantiates class before each test."""
        self.cs = ClusterSpace(self.primitive_structure, self.cutoffs, self.chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_merge_orbits(self):
        """ Tests merge orbits function"""

        # Merge orbits
        self.cs.merge_orbits({1: [2, 3], 4: [5, 6, 7, 8, 9, 10]})

        order_target = [0, 1, 2, 3]
        radii_target = [0, 0.0, 1.4142135623730951, 1.632993161855452]
        multiplicity_target = [1, 1, 21, 124]

        self.assertEqual([orb['order'] for orb in self.cs.orbit_data], order_target)
        self.assertAlmostEqualList([orb['radius'] for orb in self.cs.orbit_data], radii_target)
        self.assertEqual([orb['multiplicity'] for orb in self.cs.orbit_data], multiplicity_target)

    def test_merge_orbits_fails_for_self_merge(self):
        """ Tests that merging orbit with itself fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({2: [2, 3]})
        self.assertTrue('Cannot merge' in str(cm.exception) and 'with itself' in str(cm.exception))

    def test_merge_orbits_fails_when_merging_multiple_times(self):
        """ Tests that merging orbit multiple times fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({1: [3], 2: [3]})
        self.assertTrue('was already merged with another orbit' in str(cm.exception))

    def test_merge_orbits_fails_when_merging_different_orders(self):
        """ Tests that merging orbits with different orders fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({0: [5]})
        self.assertTrue('does not match the order' in str(cm.exception))

    def test_cluster_vectors(self):
        """ Tests that the same cluster vector is produced regardless of which orbit is
        used as key when merging """

        cv_target = [1., 0.55555556, 0.28747795, 0.13261649]

        cs_1 = self.cs.copy()
        cs_1.merge_orbits({1: [2, 3], 4: [5, 6, 7, 8, 9, 10]})
        cv_1 = cs_1.get_cluster_vector(self.structure)

        cs_2 = self.cs.copy()
        cs_2.merge_orbits({2: [1, 3], 9: [4, 5, 6, 7, 8, 10]})
        cv_2 = cs_2.get_cluster_vector(self.structure)

        self.assertAlmostEqualList(list(cv_1), cv_target)
        self.assertAlmostEqualList(list(cv_1), list(cv_2))

    def test_read_write(self):
        """Tests read/write functionality."""
        self.cs.merge_orbits({1: [2, 3], 4: [5, 6, 7, 8, 9, 10]})
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()
        self.cs.write(f.name)

        cs_read = ClusterSpace.read(f.name)
        os.remove(f.name)
        self.assertEqualAtoms(self.cs._input_structure, cs_read._input_structure)
        self.assertAlmostEqualList(list(self.cs._cutoffs), list(cs_read._cutoffs))
        self.assertEqual(self.cs._input_chemical_symbols,
                         cs_read._input_chemical_symbols)
        self.assertEqual(len(self.cs), len(cs_read))


class TestClusterSpaceMergedOrbitsTernary(unittest.TestCase):
    """
    Container for tests of the class functionality for merged orbits
    """

    def __init__(self, *args, **kwargs):
        super(TestClusterSpaceMergedOrbitsTernary, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Au', 'Cu', 'Pd']
        self.cutoffs = [5.0, 5.0]
        self.primitive_structure = bulk('Au', a=4.0)
        self.structure = self.primitive_structure.repeat((3, 3, 3))
        for i in [3, 7, 9, 14, 18, 21]:
            self.structure[i].symbol = 'Pd'

    def setUp(self):
        """Instantiates class before each test."""
        self.cs = ClusterSpace(self.primitive_structure, self.cutoffs, self.chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_merge_orbits(self):
        """ Tests merge orbits function"""

        # Merge orbits
        self.cs.merge_orbits({1: [2, 3], 4: [5, 6, 7, 8, 9, 10]}, ignore_permutations=True)

        order_target = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        radii_target = [0] * 3 + [1.4142135623730951] * 3 + [1.632993161855452] * 4
        multiplicity_target = [1, 1, 1, 21, 42, 21, 124, 372, 372, 124]

        self.assertEqual([orb['order'] for orb in self.cs.orbit_data], order_target)
        self.assertAlmostEqualList([orb['radius'] for orb in self.cs.orbit_data], radii_target)
        self.assertEqual([orb['multiplicity'] for orb in self.cs.orbit_data], multiplicity_target)

    def test_merge_orbits_fails_for_self_merge(self):
        """ Tests that merging orbit with itself fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({2: [2, 3]})
        self.assertTrue('Cannot merge' in str(cm.exception) and 'with itself' in str(cm.exception))

    def test_merge_orbits_fails_when_merging_multiple_times(self):
        """ Tests that merging orbit multiple times fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({1: [3], 2: [3]})
        self.assertTrue('was already merged with another orbit' in str(cm.exception))

    def test_merge_orbits_fails_when_merging_different_orders(self):
        """ Tests that merging orbits with different orders fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({0: [5]})
        self.assertTrue('does not match the order' in str(cm.exception))

    def test_merge_orbits_fails_when_site_permutations_differ(self):
        """Tests that merging orbits with different site permutations fails."""
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({1: [2, 3], 4: [5, 6, 7, 8, 9, 10]}, ignore_permutations=False)
        self.assertIn('Orbit 4 and orbit 5 have different site permutations', str(cm.exception))

    def test_cluster_vectors(self):
        """ Tests that the same cluster vector is produced regardless of which orbit is
        used as key when merging """

        cv_target = [1, 0.5, 0.48112522, 0.25, 0.24056261, 0.21560847, 0.125, 0.12028131,
                     0.10648148, 0.08613694]
        cs = self.cs.copy()
        cs.merge_orbits({1: [2, 3], 4: [5, 6, 7, 8, 9, 10]}, ignore_permutations=True)
        cv = cs.get_cluster_vector(self.structure)
        self.assertAlmostEqualList(list(cv), cv_target)

        cs = self.cs.copy()
        cs.merge_orbits({2: [1, 3], 4: [5, 6, 7, 8, 9, 10]}, ignore_permutations=True)
        cv = cs.get_cluster_vector(self.structure)
        self.assertAlmostEqualList(list(cv), cv_target)

        cv_target = [1, 0.5, 0.48112522, 0.25, 0.24056261, 0.21560847, 0.125, 0.12028131,
                     0.12028131, 0.10349462, 0.1124552, 0.08613694]

        cs = self.cs.copy()
        cs.merge_orbits({1: [2, 3], 5: [4, 6, 7, 8, 9, 10]}, ignore_permutations=True)
        cv = cs.get_cluster_vector(self.structure)
        self.assertAlmostEqualList(list(cv), cv_target)

    def test_read_write(self):
        """Tests read/write functionality."""
        self.cs.merge_orbits({1: [2, 3], 4: [5, 6, 7, 8, 9, 10]}, ignore_permutations=True)
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()
        self.cs.write(f.name)

        cs_read = ClusterSpace.read(f.name)
        os.remove(f.name)
        self.assertEqualAtoms(self.cs._input_structure, cs_read._input_structure)
        self.assertAlmostEqualList(list(self.cs._cutoffs), list(cs_read._cutoffs))
        self.assertEqual(self.cs._input_chemical_symbols,
                         cs_read._input_chemical_symbols)
        self.assertEqual(len(self.cs), len(cs_read))


class TestClusterSpaceMergedOrbitsSublattices(unittest.TestCase):
    """
    Container for tests of the class functionality for merged orbits
    """

    def __init__(self, *args, **kwargs):
        super(TestClusterSpaceMergedOrbitsSublattices, self).__init__(*args, **kwargs)
        self.chemical_symbols = [['H', 'X'], ['Pd', 'Au']]
        self.cutoffs = [5]
        self.primitive_structure = bulk('HPd', crystalstructure='rocksalt', a=4.0)
        self.structure = self.primitive_structure.repeat((3, 3, 3))
        for i in [8, 12, 40, 42]:
            self.structure[i].symbol = 'X'
        for i in [3, 7, 9, 15, 21]:
            self.structure[i].symbol = 'Au'

    def setUp(self):
        """Instantiates class before each test."""
        self.cs = ClusterSpace(self.primitive_structure, self.cutoffs, self.chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_merge_orbits(self):
        """ Tests merge orbits function"""

        # Merge orbits
        self.cs.merge_orbits({2: [5, 8], 3: [6, 9], 4: [7, 10]})

        order_target = [0, 1, 1, 2, 2, 2]
        radii_target = [0] * 3 + [1.0] + [1.4142135623730951] * 2
        multiplicity_target = [1, 1, 1, 38, 21, 21]

        self.assertEqual([orb['order'] for orb in self.cs.orbit_data], order_target)
        self.assertAlmostEqualList([orb['radius'] for orb in self.cs.orbit_data], radii_target)
        self.assertEqual([orb['multiplicity'] for orb in self.cs.orbit_data], multiplicity_target)

    def test_merge_orbits_fails_for_self_merge(self):
        """ Tests that merging orbit with itself fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({2: [2, 3]})
        self.assertTrue('Cannot merge' in str(cm.exception) and 'with itself' in str(cm.exception))

    def test_merge_orbits_fails_when_merging_multiple_times(self):
        """ Tests that merging orbit multiple times fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({2: [8], 5: [8]})
        self.assertTrue('was already merged with another orbit' in str(cm.exception))

    def test_merge_orbits_fails_when_merging_different_orders(self):
        """ Tests that merging orbits with different orders fails. """
        with self.assertRaises(ValueError) as cm:
            self.cs.merge_orbits({0: [5]})
        self.assertTrue('does not match the order' in str(cm.exception))

    def test_cluster_vectors(self):
        """ Tests that that the correct cluster vectors are being produced """

        cv_target = [1, 0.7037037, -0.62962963, -0.4502924, 0.49206349, 0.35802469]

        cs = self.cs.copy()
        cs.merge_orbits({2: [5, 8], 3: [6, 9], 4: [7, 10]})
        cv = cs.get_cluster_vector(self.structure)
        self.assertAlmostEqualList(list(cv), cv_target)

        cs = self.cs.copy()
        cs.merge_orbits({5: [2, 8], 6: [3, 9], 7: [4, 10]})
        cv = cs.get_cluster_vector(self.structure)
        self.assertAlmostEqualList(list(cv), cv_target)

    def test_read_write(self):
        """Tests read/write functionality."""
        self.cs.merge_orbits({2: [5, 8], 3: [6, 9], 4: [7, 10]})
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()
        self.cs.write(f.name)

        cs_read = ClusterSpace.read(f.name)
        os.remove(f.name)
        self.assertEqualAtoms(self.cs._input_structure, cs_read._input_structure)
        self.assertAlmostEqualList(list(self.cs._cutoffs), list(cs_read._cutoffs))
        self.assertEqual(self.cs._input_chemical_symbols,
                         cs_read._input_chemical_symbols)
        self.assertEqual(len(self.cs), len(cs_read))


if __name__ == '__main__':
    unittest.main()
