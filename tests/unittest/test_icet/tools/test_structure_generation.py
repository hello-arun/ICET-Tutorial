#!/usr/bin/env python3
import unittest
import numpy as np
from icet.tools.structure_generation import (_get_sqs_cluster_vector,
                                             _validate_concentrations,
                                             _concentrations_fit_structure,
                                             occupy_structure_randomly,
                                             generate_target_structure,
                                             generate_target_structure_from_supercells,
                                             generate_sqs,
                                             generate_sqs_from_supercells,
                                             generate_sqs_by_enumeration)
from tempfile import NamedTemporaryFile
from icet.input_output.logging_tools import logger, set_log_config
from icet import ClusterSpace
from ase.build import bulk
from ase import Atom


class TestStructureGenerationBinaryFCC(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationBinaryFCC, self).__init__(*args, **kwargs)
        self.prim = bulk('Au', a=4.0)
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [6.0, 5.0], ['Au', 'Pd'])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        """Test SQS cluster vector generation."""
        target_concentrations = {'Au': 0.5, 'Pd': 0.5}
        target_vector = np.array([1.0] + [0.0] * (len(self.cs) - 1))
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        self.assertTrue(np.allclose(cv, target_vector))

        target_concentrations = {'A': {'Au': 0.15, 'Pd': 0.85}}
        target_vector = np.array([1., -0.7, 0.49, 0.49, 0.49,
                                  0.49, -0.343, -0.343, -0.343,
                                  -0.343, -0.343, -0.343, -0.343])
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        self.assertTrue(np.allclose(cv, target_vector))

    def test_validate_concentrations(self):
        """Tests validation of conecntrations against cluster space."""
        concentrations = {'Au': 0.5, 'Pd': 0.5}
        _validate_concentrations(concentrations, self.cs)

        concentrations = {'Au': 0.1, 'Pd': 0.7}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('Concentrations must sum up to 1', str(cm.exception))

        concentrations = {'Au': 0.1, 'Pd': 0.8, 'Cu': 0.1}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('not the same as those in the specified', str(cm.exception))

        concentrations = {'Au': 1.0}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('not the same as those in the specified', str(cm.exception))

    def test_concentrations_fit_structure(self):
        """
        Tests check of concentrations against an ASE Atoms object
        belonging to a cluster space
        """
        concentrations = {'A': {'Au': 1 / 3, 'Pd': 2 / 3}}
        self.assertTrue(_concentrations_fit_structure(
            self.supercell, self.cs, concentrations))

        concentrations = {'A': {'Au': 0.5, 'Pd': 0.5}}
        self.assertFalse(_concentrations_fit_structure(
            self.supercell, self.cs, concentrations))

    def test_occupy_structure_randomly(self):
        """Tests random occupation of ASE Atoms object"""
        structure = self.prim.repeat(2)
        target_concentrations = {'Au': 0.5, 'Pd': 0.5}
        occupy_structure_randomly(structure, self.cs,
                                  target_concentrations)
        syms = structure.get_chemical_symbols()
        self.assertEqual(syms.count('Au'), len(structure) // 2)

        structure = self.prim.repeat(3)
        target_concentrations = {'Au': 1 / 3, 'Pd': 2 / 3}
        occupy_structure_randomly(structure, self.cs,
                                  target_concentrations)
        syms = structure.get_chemical_symbols()
        self.assertEqual(syms.count('Au'), len(structure) // 3)
        self.assertEqual(syms.count('Pd'), 2 * len(structure) // 3)

    def test_generate_target_structure(self):
        """Test generation of a structure based on a target cluster vector"""
        # Exact target vector from 2 atoms cell
        # target_cv = np.array([1., 0., 0., -1., 0., 1.,
        #                      0., 0., 0., 0., 0., 0., 0.])
        target_cv = np.array([1.,  0., -1 / 3,  1., -1 / 3,
                              1.,  0.,  0.,  0.,  0.,
                              0.,  0.,  0.])
        target_conc = {'Au': 0.5, 'Pd': 0.5}

        kwargs = dict(cluster_space=self.cs,
                      max_size=4,
                      target_concentrations=target_conc,
                      target_cluster_vector=target_cv,
                      n_steps=500,
                      random_seed=42,
                      optimality_weight=0.3)

        # This should be simple enough to always work
        structure = generate_target_structure(**kwargs)
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

        # Using include_smaller_cells = False
        structure = generate_target_structure(**kwargs, include_smaller_cells=False)
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

        # Using non-pbc
        structure = generate_target_structure(**kwargs, include_smaller_cells=False,
                                              pbc=(True, False, False))
        target_cell = [[0, 8, 8], [2, 0, 2], [2, 2, 0]]
        self.assertTrue(np.allclose(structure.cell, target_cell))
        target_cv = [1., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

    def test_generate_target_from_supercells(self):
        """Test generation of a structure based on a target cluster vector and a list
        of supercells"""
        target_cv = [1., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
        target_conc = {'Au': 0.5, 'Pd': 0.5}
        kwargs = dict(cluster_space=self.cs,
                      target_concentrations=target_conc,
                      target_cluster_vector=target_cv,
                      n_steps=500,
                      random_seed=42,
                      optimality_weight=0.3)

        supercells = [self.prim.repeat((2, 2, 1)), self.prim.repeat((2, 1, 1))]
        # This should be simple enough to always work
        structure = generate_target_structure_from_supercells(supercells=supercells, **kwargs)
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

        # Log output to StringIO stream
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        # Use supercells that do not fit
        supercells = [self.prim.repeat((2, 2, 1)), self.prim.repeat((3, 1, 1))]
        logfile = NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
        set_log_config(filename=logfile.name)
        structure = generate_target_structure_from_supercells(supercells=supercells, **kwargs)
        logfile.seek(0)
        lines = logfile.readlines()
        logfile.close()
        self.assertIn('At least one supercell was not commensurate', lines[0])
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

        # Use two supercells that do not fit
        supercells = [self.prim.repeat((3, 3, 1)), self.prim.repeat((3, 1, 1))]
        logfile = NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
        set_log_config(filename=logfile.name)
        with self.assertRaises(ValueError) as cm:
            generate_target_structure_from_supercells(supercells=supercells, **kwargs)
        logfile.seek(0)
        lines = logfile.readlines()
        logfile.close()
        self.assertEqual(len(lines), 1)  # Warning should be issued once
        self.assertIn('At least one supercell was not commensurate', lines[0])
        self.assertIn('No supercells that may host the specified', str(cm.exception))

    def test_generate_sqs(self):
        """Test generation of SQS structure"""

        kwargs = dict(cluster_space=self.cs,
                      max_size=4,
                      target_concentrations={'Au': 0.5, 'Pd': 0.5},
                      n_steps=500,
                      random_seed=42,
                      optimality_weight=0.0)

        # This should be simple enough to always work
        structure = generate_sqs(**kwargs)
        target_cv = [1., 0., -0.16666667, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

        # Using-non pbc
        structure = generate_sqs(**kwargs, pbc=(False, True, False))
        target_cell = [[0, 2, 2], [8, 0, 8], [2, 2, 0]]
        self.assertTrue(np.allclose(structure.cell, target_cell))
        target_cv = [1., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

    def test_generate_sqs_from_supercells(self):
        """Test generation of SQS structure from list of supercells"""
        target_conc = {'Au': 0.5, 'Pd': 0.5}
        kwargs = dict(cluster_space=self.cs,
                      target_concentrations=target_conc,
                      n_steps=500,
                      random_seed=42,
                      optimality_weight=0.0)

        supercells = [self.prim.repeat((2, 2, 1)), self.prim.repeat((2, 1, 1))]
        structure = generate_sqs_from_supercells(supercells=supercells, **kwargs)
        target_cv = [1., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

        # Log output to StringIO stream
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        # Test with supercell that does not match
        supercells = [self.prim]
        logfile = NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
        set_log_config(filename=logfile.name)
        with self.assertRaises(ValueError) as cm:
            generate_sqs_from_supercells(supercells=supercells, **kwargs)
        logfile.seek(0)
        lines = logfile.readlines()
        logfile.close()
        self.assertEqual(len(lines), 1)
        self.assertIn('At least one supercell was not commensurate', lines[0])
        self.assertIn('No supercells that may host the specified', str(cm.exception))

    def test_generate_sqs_by_enumeration(self):
        """Test generation of SQS structure"""
        kwargs = dict(cluster_space=self.cs,
                      max_size=4,
                      target_concentrations={'Au': 0.5, 'Pd': 0.5},
                      optimality_weight=0.0)

        structure = generate_sqs_by_enumeration(**kwargs)
        target_cv = [1., 0., -0.16666667, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))

        # Using non-pbc
        structure = generate_sqs(**kwargs, pbc=(False, True, False))
        target_cell = [[0, 2, 2], [8, 0, 8], [2, 2, 0]]
        self.assertTrue(np.allclose(structure.cell, target_cell))
        target_cv = [1., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(self.cs.get_cluster_vector(structure), target_cv))


class TestStructureGenerationTernaryFCC(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationTernaryFCC,
              self).__init__(*args, **kwargs)
        self.prim = bulk('Au', a=4.0)
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [5.0, 4.0], ['Au', 'Pd', 'Cu'])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        """Test SQS cluster vector generation."""
        target_concentrations = {'Au': 0.5, 'Pd': 0.3, 'Cu': 0.2}
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        target_vector = [1., 0.2, 0.17320508, 0.04, 0.03464102, 0.03,
                         0.04, 0.03464102, 0.03, 0.04, 0.03464102, 0.03,
                         0.008, 0.0069282, 0.006, 0.00519615, 0.008, 0.0069282,
                         0.0069282, 0.006, 0.006, 0.00519615]
        self.assertTrue(np.allclose(cv, target_vector))

    def test_validate_concentrations(self):
        """Tests validation of conecntrations against cluster space."""
        concentrations = {'Au': 0.1, 'Pd': 0.8, 'Cu': 0.1}
        _validate_concentrations(concentrations, self.cs)

        concentrations = {'Au': 0.1, 'Pd': 0.7, 'Cu': 0.05}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('Concentrations must sum up to 1', str(cm.exception))

        concentrations = {'Au': 0.5, 'Pd': 0.5}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('not the same as those in the specified', str(cm.exception))

    def test_concentrations_fit_structure(self):
        """
        Tests check of concentrations against an ASE Atoms object
        belonging to a cluster space
        """
        concentrations = {'A': {'Au': 1 / 3, 'Pd': 1 / 3, 'Cu': 1 / 3}}
        self.assertTrue(_concentrations_fit_structure(
            self.supercell, self.cs, concentrations))

        concentrations = {'A': {'Au': 0.5, 'Pd': 0.5, 'Cu': 0.0}}
        self.assertFalse(_concentrations_fit_structure(
            self.supercell, self.cs, concentrations))

    def test_occupy_structure_randomly(self):
        """Tests random occupation of structure object"""
        structure = self.prim.repeat(2)
        target_concentrations = {'Cu': 0.25, 'Au': 0.25, 'Pd': 0.5}
        occupy_structure_randomly(structure, self.cs,
                                  target_concentrations)
        syms = structure.get_chemical_symbols()
        self.assertEqual(syms.count('Cu'), len(structure) // 4)
        self.assertEqual(syms.count('Au'), len(structure) // 4)
        self.assertEqual(syms.count('Pd'), len(structure) // 2)


class TestStructureGenerationHCP(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationHCP,
              self).__init__(*args, **kwargs)
        self.prim = bulk('Au', a=4.0, crystalstructure='hcp')
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [5.0, 4.0], ['Au', 'Pd'])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        """Test SQS cluster vector generation."""
        target_concentrations = {'Au': 0.5, 'Pd': 0.5}
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        target_vector = np.zeros(len(self.cs))
        target_vector[0] = 1.0
        self.assertTrue(np.allclose(cv, target_vector))

    def test_validate_concentrations(self):
        """Tests validation of conecntrations against cluster space."""

        # Test with dict
        concentrations = {'Au': 0.1, 'Pd': 0.9}
        ret_conc = _validate_concentrations(concentrations, self.cs)
        self.assertIsInstance(ret_conc, dict)
        self.assertEqual(len(ret_conc), 1)
        self.assertIsInstance(ret_conc['A'], dict)
        self.assertEqual(len(ret_conc['A']), 2)
        for el in concentrations.keys():
            self.assertIn(el, ret_conc['A'])
            self.assertAlmostEqual(ret_conc['A'][el], concentrations[el])

        # Test with list of dicts
        list_concentrations = {'A': concentrations}
        ret_conc = _validate_concentrations(list_concentrations, self.cs)
        self.assertIsInstance(ret_conc, dict)
        self.assertEqual(len(ret_conc), 1)
        self.assertIsInstance(ret_conc['A'], dict)
        self.assertEqual(len(ret_conc['A']), 2)
        for el in concentrations.keys():
            self.assertIn(el, ret_conc['A'])
            self.assertAlmostEqual(ret_conc['A'][el], concentrations[el])

        concentrations = {'Au': 0.1, 'Pd': 0.7}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('Concentrations must sum up to 1', str(cm.exception))

        concentrations = {'Au': 1.0}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('not the same as those in the specified', str(cm.exception))

    def test_concentrations_fit_structure(self):
        """
        Tests check of concentrations against an ASE Atoms object
        belonging to a cluster space
        """
        concentrations = {'A': {'Au': 1 / 3, 'Pd': 2 / 3}}
        self.assertTrue(_concentrations_fit_structure(
            self.supercell, self.cs, concentrations))

        concentrations = {'A': {'Au': 3 / 5, 'Pd': 2 / 5}}
        self.assertFalse(_concentrations_fit_structure(
            self.supercell, self.cs, concentrations))

    def test_occupy_structure_randomly(self):
        """Tests random occupation of ASE Atoms object"""
        structure = self.prim.repeat(3)
        target_concentrations = {'Pd': 1 / 3, 'Au': 2 / 3}
        occupy_structure_randomly(structure, self.cs,
                                  target_concentrations)
        syms = structure.get_chemical_symbols()
        self.assertEqual(syms.count('Au'), 2 * len(structure) // 3)
        self.assertEqual(syms.count('Pd'), len(structure) // 3)


class TestStructureGenerationSublatticesFCC(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationSublatticesFCC,
              self).__init__(*args, **kwargs)
        self.prim = bulk('Au', a=4.0)
        self.prim.append(Atom('H', position=(2.0, 2.0, 2.0)))
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [5.0, 4.0], [
                               ['Au', 'Pd', 'Cu'], ['H', 'V']])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        """Test SQS cluster vector generation."""
        target_concentrations = {'A': {'Au': 0.4, 'Pd': 0.2, 'Cu': 0.4},
                                 'B': {'H': 0.5, 'V': 0.5}}
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        target_vector = [1., -0.1, 0.17320508, 0., 0., 0.,
                         0.01, -0.01732051, 0.03, 0., 0., 0.,
                         0.01, -0.01732051, 0.03, 0., 0., 0.,
                         0.01, -0.01732051, 0.03, 0., 0., 0.,
                         0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0.,
                         0., 0., -0.001, 0.00173205, -0.003, 0.00519615,
                         0., -0.001, 0.00173205, 0.00173205, -0.003, -0.003,
                         0.00519615, 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(cv, target_vector))

    def test_validate_concentrations(self):
        """Tests validation of conecntrations against cluster space."""
        concentrations = {'A': {'Au': 0.2, 'Pd': 0.6, 'Cu': 0.2}, 'B': {'H': 0.8, 'V': 0.2}}
        ret_conc = _validate_concentrations(concentrations, self.cs)
        self.assertIsInstance(ret_conc, dict)
        self.assertEqual(len(ret_conc), 2)
        self.assertIsInstance(ret_conc['A'], dict)

        concentrations = {'A': {'Au': 0.1, 'Pd': 0.7, 'Cu': 0.2}, 'B': {'H': 0.0, 'V': 0.9}}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('Concentrations must sum up to 1', str(cm.exception))

        concentrations = {'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'Cu': 1}}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('not the same as those in the specified', str(cm.exception))

        concentrations = {'Au': 2 / 6, 'Pd': 1 / 6, 'Cu': 3 / 6}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn("A sublattice (B: ['H', 'V']) is missing",
                      str(cm.exception))

    def test_concentrations_fit_structure(self):
        """
        Tests check of concentrations against an ASE Atoms object
        belonging to a cluster space
        """
        concentrations = {'A': {'Au': 1 / 3, 'Pd': 1 / 3, 'Cu': 1 / 3},
                          'B': {'H': 2 / 3, 'V': 1 / 3}}
        self.assertTrue(_concentrations_fit_structure(
            self.supercell, self.cs, concentrations))

        concentrations = {'A': {'Au': 1 / 2, 'Pd': 1 / 4, 'Cu': 1 / 4},
                          'B': {'H': 2 / 3, 'V': 1 / 3}}
        self.assertFalse(_concentrations_fit_structure(
            self.supercell, self.cs, concentrations))

    def test_occupy_structure_randomly(self):
        """Tests random occupation of ASE Atoms object"""
        structure = self.prim.repeat(2)
        target_concentrations = {'A': {'Cu': 1 / 4, 'Au': 2 / 4, 'Pd': 1 / 4},
                                 'B': {'H': 3 / 4, 'V': 1 / 4}}
        occupy_structure_randomly(structure, self.cs,
                                  target_concentrations)
        syms = structure.get_chemical_symbols()
        self.assertEqual(syms.count('Cu'), len(structure) // 8)
        self.assertEqual(syms.count('Au'), len(structure) // 4)
        self.assertEqual(syms.count('Pd'), len(structure) // 8)
        self.assertEqual(syms.count('H'), 3 * len(structure) // 8)
        self.assertEqual(syms.count('V'), len(structure) // 8)

    def test_generate_sqs_by_enumeration(self):
        """Test generation of SQS structure"""

        target_conc = {'A': {'Cu': 1 / 4, 'Au': 2 / 4, 'Pd': 1 / 4},
                       'B': {'H': 3 / 4, 'V': 1 / 4}}
        structure = generate_sqs_by_enumeration(cluster_space=self.cs,
                                                max_size=4,
                                                include_smaller_cells=False,
                                                target_concentrations=target_conc,
                                                optimality_weight=1.0)
        target_cv = [1.00000000e+00,  1.25000000e-01,  2.16506351e-01, -5.00000000e-01,
                     -1.25000000e-01, -7.21687836e-02, -3.12500000e-02, -1.85037171e-17,
                     -3.12500000e-02,  1.66666667e-01,  3.12500000e-02, -1.62379763e-01,
                     -1.25000000e-01,  1.08253175e-01, -3.70074342e-17,  0.00000000e+00,
                     -6.25000000e-02, -1.08253175e-01,  1.56250000e-02,  2.70632939e-02,
                     4.68750000e-02,  2.50000000e-01, -9.25185854e-18,  1.80421959e-02,
                     6.25000000e-02,  8.33333333e-02, -3.70074342e-17, -2.22044605e-16,
                     -2.52590743e-01, -1.25000000e-01,  1.25000000e-01, -7.21687836e-02,
                     -1.56250000e-02,  4.51054898e-02, -8.11898816e-02,  4.68750000e-02,
                     5.20833333e-02,  1.80421959e-02, -1.56250000e-02, -1.35316469e-02,
                     -1.56250000e-02, -4.05949408e-02,  0.00000000e+00, -6.25000000e-02,
                     2.54426110e-17, -5.41265877e-02,  3.12500000e-02, -3.23815049e-17,
                     -5.41265877e-02,  1.66666667e-01,  1.56250000e-01, -8.78926561e-17,
                     -9.37500000e-02, -1.87500000e-01,  1.08253175e-01]
        self.assertTrue(np.allclose(
            self.cs.get_cluster_vector(structure), target_cv))


class TestStructureGenerationInactiveSublatticeFCC(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationInactiveSublatticeFCC, self).__init__(*args, **kwargs)
        self.prim = bulk('Au', a=4.0)
        self.cs_with_only_active_sl = ClusterSpace(self.prim, [6.0, 5.0], ['Au', 'Pd'])
        self.prim.append(Atom('H', position=(2.0, 2.0, 2.0)))
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [6.0, 5.0], [['Au', 'Pd'], ['H']])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        """Test SQS cluster vector generation."""
        # It should be the same as for cs without inactive lattice
        target_concentrations = {'Au': 0.7, 'Pd': 0.3}
        target_cv = _get_sqs_cluster_vector(self.cs_with_only_active_sl, target_concentrations)
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        self.assertTrue(np.allclose(cv, target_cv))

    def test_validate_concentrations(self):
        """Tests validation of conecntrations against cluster space."""
        concentrations = {'Au': 0.4, 'Pd': 0.6}
        ret_conc = _validate_concentrations(concentrations, self.cs)
        self.assertIsInstance(ret_conc, dict)
        self.assertEqual(len(ret_conc), 1)
        self.assertIsInstance(ret_conc['A'], dict)

        concentrations = {'A': {'Au': 0.4, 'Pd': 0.6}, 'B': {'H': 1.0}}
        ret_conc = _validate_concentrations(concentrations, self.cs)
        self.assertIsInstance(ret_conc, dict)
        self.assertEqual(len(ret_conc), 2)
        self.assertIsInstance(ret_conc['A'], dict)

        concentrations = {'Au': 0.1, 'Pd': 0.8}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('Concentrations must sum up to 1', str(cm.exception))

        concentrations = {'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'C': 1}}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('not the same as those in the specified', str(cm.exception))

    def test_occupy_structure_randomly(self):
        """Tests random occupation of ASE Atoms object"""
        structure = self.prim.repeat(2)
        target_concentrations = {'Au': 3 / 4, 'Pd': 1 / 4}
        occupy_structure_randomly(structure, self.cs,
                                  target_concentrations)
        syms = structure.get_chemical_symbols()
        self.assertEqual(syms.count('Au'), 3 * (len(structure) // 2) // 4)
        self.assertEqual(syms.count('Pd'), (len(structure) // 2) // 4)

    def test_generate_sqs_by_enumeration(self):
        """Test generation of SQS structure"""

        target_conc = {'Pd': 1 / 2, 'Au': 1 / 2}
        structure = generate_sqs_by_enumeration(cluster_space=self.cs,
                                                max_size=4,
                                                target_concentrations=target_conc,
                                                optimality_weight=0.0)
        target_cv = [1., 0., -0.16666667, 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(
            self.cs.get_cluster_vector(structure), target_cv))

        target_conc = {'Pd': 1 / 3, 'Au': 2 / 3}
        target_cv = [1., 0., -0.16666667, 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0.]
        target_cv = [1, 1/3, -1/9, 1/9, 1/3, -1/9,
                     -1/3, -1/9, 1/9, -1/9, 1/9, 1/3, -1/3]
        structure = generate_sqs_by_enumeration(cluster_space=self.cs,
                                                max_size=3,
                                                target_concentrations=target_conc,
                                                optimality_weight=0.0,
                                                pbc=True)
        self.assertTrue(np.allclose(
            self.cs.get_cluster_vector(structure), target_cv))


class TestStructureGenerationInactiveSublatticeSameSpeciesFCC(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationInactiveSublatticeSameSpeciesFCC, self).__init__(
            *args, **kwargs)
        self.prim = bulk('Au', a=4.0)
        self.cs_with_only_active_sl = ClusterSpace(self.prim, [6.0, 5.0], ['Au', 'Pd'])
        self.prim.append(Atom('Au', position=(2.0, 2.0, 2.0)))
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [6.0, 5.0], [['Au', 'Pd'], ['Au']])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        """Test SQS cluster vector generation."""
        target_concentrations = {'Au': 0.7, 'Pd': 0.3}

        # It should be the same as for cs without inactive lattice
        target_cv = _get_sqs_cluster_vector(self.cs_with_only_active_sl, target_concentrations)
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        self.assertTrue(np.allclose(cv, target_cv))

        # Test also when inactive sublattice is specified
        target_concentrations = {'A': {'Au': 0.7, 'Pd': 0.3}, 'B': {'Au': 1}}
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        self.assertTrue(np.allclose(cv, target_cv))

        # It should be the same as for cs without inactive lattice
        target_cv = _get_sqs_cluster_vector(self.cs_with_only_active_sl, target_concentrations)
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        self.assertTrue(np.allclose(cv, target_cv))

    def test_validate_concentrations(self):
        """Tests validation of conecntrations against cluster space."""
        concentrations = {'Au': 0.4, 'Pd': 0.6}
        ret_conc = _validate_concentrations(concentrations, self.cs)
        self.assertIsInstance(ret_conc, dict)
        self.assertEqual(len(ret_conc), 1)
        self.assertIsInstance(ret_conc['A'], dict)

        concentrations = {'Au': 0.1, 'Pd': 0.8}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('Concentrations must sum up to 1', str(cm.exception))

        concentrations = {'A': {'Au': 0.4, 'Pd': 0.6}, 'B': {'Au': 1.0}}
        ret_conc = _validate_concentrations(concentrations, self.cs)
        self.assertIsInstance(ret_conc, dict)
        self.assertEqual(len(ret_conc), 2)
        self.assertIsInstance(ret_conc['A'], dict)

        concentrations = {'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'C': 1}}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertIn('not the same as those in the specified', str(cm.exception))

    def test_occupy_structure_randomly(self):
        """Tests random occupation of ASE Atoms object"""
        structure = self.prim.repeat(2)
        target_concentrations = {'Au': 3 / 4, 'Pd': 1 / 4}
        occupy_structure_randomly(structure, self.cs,
                                  target_concentrations)
        syms = structure.get_chemical_symbols()
        self.assertEqual(syms.count('Au'), 3 * (len(structure) // 2) // 4 + len(structure) // 2)
        self.assertEqual(syms.count('Pd'), (len(structure) // 2) // 4)

    def test_generate_sqs_by_enumeration(self):
        """Test generation of SQS structure"""

        target_conc = {'Pd': 1 / 2, 'Au': 1 / 2}
        structure = generate_sqs_by_enumeration(cluster_space=self.cs,
                                                max_size=4,
                                                target_concentrations=target_conc,
                                                optimality_weight=0.0)
        target_cv = [1., 0., -0.16666667, 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(
            self.cs.get_cluster_vector(structure), target_cv))


if __name__ == '__main__':
    unittest.main()
