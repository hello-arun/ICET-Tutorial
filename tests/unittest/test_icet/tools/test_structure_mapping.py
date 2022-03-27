#!/usr/bin/env python3

import numpy as np
import unittest
from tempfile import NamedTemporaryFile

from icet.tools import map_structure_to_reference
from icet.tools.structure_mapping import (_get_reference_supercell,
                                          _match_positions,
                                          calculate_strain_tensor)
from icet.input_output.logging_tools import logger, set_log_config
from ase import Atom
from ase.build import bulk


class TestStructureMapping(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        from ase.build import bulk

        reference = bulk('Au', a=4.0)
        reference.append(Atom('H', (2, 2, 2)))
        self.reference = reference

        structure = reference.repeat(2)
        for i in [0, 1, 2, 3, 4, 6, 7]:
            if structure[i].symbol == 'Au':
                structure[i].symbol = 'Pd'
            elif structure[i].symbol == 'H':
                del structure[i]

        # Displace the atoms somewhat
        rattle = [[0.147, -0.037, -0.01],
                  [-0.089, 0.084, -0.063],
                  [0.256, -0.037, 0.097],
                  [-0.048, 0.005, -0.093],
                  [-0.159, -0.194, -0.03],
                  [0.004, -0.041, -0.003],
                  [-0.015, -0.014, -0.007],
                  [-0.023, 0.094, -0.024],
                  [-0.01, 0.075, -0.075],
                  [0.029, -0.024, 0.079],
                  [0.105, 0.172, -0.147]]

        structure.positions = structure.positions + rattle
        structure.set_cell(structure.cell * 1.01, scale_atoms=True)
        self.structure = structure

        super(TestStructureMapping, self).__init__(*args, **kwargs)

    def test_get_reference_supercell(self):
        """
        Tests that retrieval of a reference supercell works.
        """
        supercell, P = _get_reference_supercell(self.structure, self.reference)
        target_cell = np.array([[0., 4., 4.],
                                [4., 0., 4.],
                                [4., 4., 0.]])
        target_P = np.array([[2., 0., 0.],
                             [0., 2., 0.],
                             [0., 0., 2.]])
        target_formula = 'H8Au8'
        self.assertTrue(np.allclose(supercell.cell, target_cell))
        self.assertEqual(supercell.get_chemical_formula(), target_formula)
        self.assertTrue(np.allclose(P, target_P))

        # Should work with default tol_cell when inert_species are specified
        supercell, P = _get_reference_supercell(self.structure, self.reference,
                                                inert_species=['Au', 'Pd'])
        self.assertTrue(np.allclose(supercell.cell, target_cell))
        self.assertEqual(supercell.get_chemical_formula(), target_formula)

        # Assuming no cell relaxation, proper match
        structure_scaled = self.structure.copy()
        structure_scaled.set_cell(structure_scaled.cell / 1.01, scale_atoms=True)
        supercell, P = _get_reference_supercell(structure_scaled, self.reference,
                                                assume_no_cell_relaxation=True)
        self.assertTrue(np.allclose(supercell.cell, target_cell))
        self.assertEqual(supercell.get_chemical_formula(), target_formula)

        # Mismatch in boundary conditions
        structure_nopbc = self.structure.copy()
        structure_nopbc.set_pbc([True, True, False])
        with self.assertRaises(ValueError) as context:
            _get_reference_supercell(structure_nopbc, self.reference)
        self.assertIn('The boundary conditions of', str(context.exception))

    def test_match_positions(self):
        """
        Tests that the final step in mapping works.
        """
        # Match 1 atom structure
        mapped, drmax, dravg, warning = _match_positions(bulk('Au'), bulk('Au'))
        self.assertEqual(len(mapped), 1)
        self.assertAlmostEqual(drmax, 0)
        self.assertAlmostEqual(dravg, 0)
        self.assertTrue(warning is None)

        # Mismatching cell metrics
        with self.assertRaises(ValueError) as context:
            _match_positions(self.structure, self.reference.repeat(2))
        self.assertIn('The cell metrics of reference and relaxed',
                      str(context.exception))

        # Mismatching cell metrics, too many atoms in relaxed cell
        reference = self.reference.repeat(2)
        reference.set_cell(reference.cell * 1.01, scale_atoms=True)
        structure = self.structure.copy()
        for x in np.linspace(0, 1, 10):
            structure.append(Atom('Au', position=(x, 0, 0)))
        with self.assertRaises(ValueError) as context:
            _match_positions(structure, reference)
        self.assertIn('The relaxed structure contains more atoms than the reference',
                      str(context.exception))

        # Mismatching boundary conditions
        reference = self.reference.repeat(2)
        reference.set_cell(reference.cell * 1.01, scale_atoms=True)
        reference.set_pbc([True, False, False])
        with self.assertRaises(ValueError) as context:
            _match_positions(structure, reference)
        self.assertIn('The boundary conditions of', str(context.exception))

        # Working example: volume change
        reference = self.reference.repeat(2)
        reference.set_cell(reference.cell * 1.01, scale_atoms=True)
        mapped, drmax, dravg, warning = _match_positions(self.structure, reference)
        self.assertAlmostEqual(drmax, 0.279012386)
        self.assertAlmostEqual(dravg, 0.140424392)
        self.assertEqual(mapped.get_chemical_formula(), 'H3Au6Pd2X5')
        self.assertEqual(warning, None)

        # check attached arrays
        target_val = [[-0.14847, 0.03737, 0.01010],
                      [None, None, None],
                      [0.08989, -0.08484, 0.06363],
                      [None, None, None],
                      [-0.25856, 0.03737, -0.09797],
                      [None, None, None],
                      [0.04848, -0.00505, 0.09393],
                      [None, None, None],
                      [0.16059, 0.19594, 0.03030],
                      [-0.00404, 0.04141, 0.00303],
                      [0.01515, 0.01414, 0.00707],
                      [None, None, None],
                      [0.02323, -0.09494, 0.02424],
                      [0.01010, -0.07575, 0.07575],
                      [-0.02929, 0.02424, -0.07979],
                      [-0.10605, -0.17372, 0.14847]]
        for a, t in zip(mapped.arrays['Displacement'], target_val):
            if t[0] is None:
                continue
            self.assertTrue(np.allclose(a, t))
        target_val = [[0.15343359, 1.87193031, 1.98820700],
                      [None, None, None],
                      [0.13902091, 1.93302127, 1.93829131],
                      [None, None, None],
                      [0.27901239, 1.76455816, 1.93970336],
                      [None, None, None],
                      [0.10582371, 1.92668665, 1.97376277],
                      [None, None, None],
                      [0.25514647, 1.83136619, 1.86995083],
                      [0.04171679, 1.97859644, 2.01638753],
                      [0.02189628, 2.00491233, 2.00592967],
                      [None, None, None],
                      [0.10070161, 1.92535275, 1.99815195],
                      [0.10760174, 1.94575130, 1.94575130],
                      [0.08838510, 1.94058247, 1.99245585],
                      [0.25192972, 1.85527351, 1.88256468]]
        for a, t in zip(mapped.arrays['Minimum_Distances'], target_val):
            if t[0] is None:
                continue
            self.assertTrue(np.allclose(a, t))

        # Working example: with less than 3 atoms
        reference = self.reference.copy()
        structure = self.reference.copy()
        structure[0].position += [0, 0, 0.1]
        mapped, drmax, dravg, warning = _match_positions(structure, reference)
        self.assertAlmostEqual(drmax, 0.1)
        self.assertAlmostEqual(dravg, 0.05)
        self.assertEqual(mapped.get_chemical_formula(), 'HAu')
        self.assertTrue(np.allclose(mapped.arrays['Displacement'], [[0, 0, -0.1], [0, 0, 0]]))
        self.assertTrue(np.allclose(mapped.arrays['Displacement_Magnitude'], [0.1, 0]))
        self.assertTrue(np.allclose(mapped.arrays['Minimum_Distances'], [[0.1, 1.9], [0, 2]]))
        self.assertEqual(warning, None)

    def test_map_structure_to_reference(self):
        """
        Tests that mapping algorithm wrapper works.
        """
        def test_mapping(structure,
                         reference=None,
                         expected_drmax=0.276249887,
                         expected_dravg=0.139034051,
                         expected_chemical_formula='H3Au6Pd2X5',
                         **kwargs):
            """
            Convenience wrapper for testing mapping.
            """
            if reference is None:
                reference = self.reference
            logfile = NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
            set_log_config(filename=logfile.name)
            mapped, info = map_structure_to_reference(structure,
                                                      reference,
                                                      **kwargs)
            self.assertEqual(len(info), 7)
            self.assertAlmostEqual(info['drmax'], expected_drmax)
            self.assertAlmostEqual(info['dravg'], expected_dravg)
            self.assertEqual(mapped.get_chemical_formula(), expected_chemical_formula)
            logfile.seek(0)
            lines = logfile.readlines()
            logfile.close()
            return lines, info

        # Log ClusterSpace output to StringIO stream
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        # Standard, warning-free mapping
        logfile_lines, info = test_mapping(self.structure, inert_species=['Au', 'Pd'])
        self.assertEqual(len(logfile_lines), 0)
        self.assertEqual(len(info['warnings']), 0)

        # Warn when there is a lot of volumetric strain
        structure = self.structure.copy()
        structure.set_cell(1.2 * structure.cell, scale_atoms=True)
        logfile_lines, info = test_mapping(structure, inert_species=['Au', 'Pd'])
        self.assertEqual(len(logfile_lines), 1)
        self.assertIn('High volumetric strain', logfile_lines[0])
        self.assertIn('high_volumetric_strain', info['warnings'])

        # Do not warn if warnings are suppressed
        structure = self.structure.copy()
        structure.set_cell(1.2 * structure.cell, scale_atoms=True)
        logfile_lines, info = test_mapping(structure, inert_species=['Au', 'Pd'],
                                           suppress_warnings=True)
        self.assertEqual(len(logfile_lines), 0)

        # Warning-free assuming no cell relaxation
        structure = self.structure.copy()
        structure.set_cell((1 / 1.01) * structure.cell, scale_atoms=True)
        logfile_lines, info = test_mapping(structure, assume_no_cell_relaxation=True)
        self.assertEqual(len(logfile_lines), 0)
        self.assertEqual(len(info['warnings']), 0)

        # Warning even with little strain with no cell relaxation assumption
        structure = self.structure.copy()
        structure.set_cell(structure.cell, scale_atoms=True)
        logfile_lines, info = test_mapping(structure, assume_no_cell_relaxation=True)
        self.assertEqual(len(logfile_lines), 1)
        self.assertIn('High volumetric strain', logfile_lines[0])
        self.assertIn('high_volumetric_strain', info['warnings'])

        # Anisotropic strain
        structure = self.structure.copy()
        A = [[1.2, 0, 0], [0, 1 / 1.2, 0], [0, 0, 1.]]
        structure.set_cell(np.dot(structure.cell, A), scale_atoms=True)
        logfile_lines, info = test_mapping(structure, inert_species=['Au', 'Pd'])
        self.assertEqual(len(logfile_lines), 1)
        self.assertIn('High anisotropic strain', logfile_lines[0])
        self.assertIn('high_anisotropic_strain', info['warnings'])

        # Test warnings when two atoms are close to one site
        reference = bulk('Au', a=4.0, crystalstructure='sc').repeat((3, 1, 1))
        structure = reference.copy()
        structure[1].position = structure[0].position + np.array([0.1, 0, 0])
        logfile_lines, info = test_mapping(structure,
                                           reference=reference,
                                           expected_chemical_formula='Au3',
                                           expected_drmax=3.9,
                                           expected_dravg=3.9 / 3)
        self.assertEqual(len(logfile_lines), 3)
        self.assertIn('An atom was mapped to a site that was further away', logfile_lines[0])
        self.assertIn('Large maximum relaxation distance', logfile_lines[1])
        self.assertIn('Large average relaxation distance', logfile_lines[2])
        self.assertIn('large_average_relaxation_distance', info['warnings'])
        self.assertIn('large_average_relaxation_distance', info['warnings'])
        self.assertIn('possible_ambiguity_in_mapping', info['warnings'])

        # Test warnings when an atom is close to two sites
        reference = bulk('Au', a=4.0, crystalstructure='sc').repeat((3, 1, 1))
        structure = reference.copy()
        structure[1].position += np.array([2.0, 0, 0])
        logfile_lines, info = test_mapping(structure,
                                           reference=reference,
                                           expected_chemical_formula='Au3',
                                           expected_drmax=2.0,
                                           expected_dravg=2.0 / 3)
        self.assertEqual(len(logfile_lines), 3)
        self.assertIn('An atom was approximately equally far from its two', logfile_lines[0])
        self.assertIn('Large maximum relaxation distance', logfile_lines[1])
        self.assertIn('Large average relaxation distance', logfile_lines[2])
        self.assertIn('large_average_relaxation_distance', info['warnings'])
        self.assertIn('large_average_relaxation_distance', info['warnings'])
        self.assertIn('possible_ambiguity_in_mapping', info['warnings'])

        # Large deviations
        structure = self.structure.copy()
        structure.positions += [1, 0, 0]
        logfile_lines, info = test_mapping(structure, inert_species=['Au', 'Pd'],
                                           expected_drmax=1.11822844,
                                           expected_dravg=0.95130331)
        self.assertEqual(len(logfile_lines), 8)
        self.assertIn('Large maximum relaxation distance', logfile_lines[6])
        self.assertIn('Large average relaxation distance', logfile_lines[7])
        self.assertIn('large_maximum_relaxation_distance', info['warnings'])
        self.assertIn('large_average_relaxation_distance', info['warnings'])
        self.assertIn('large_average_relaxation_distance', info['warnings'])
        self.assertIn('possible_ambiguity_in_mapping', info['warnings'])

        # Match 1 atom structure
        logfile_lines, info = test_mapping(bulk('Au'), reference=bulk('Au'),
                                           expected_dravg=0, expected_drmax=0,
                                           expected_chemical_formula='Au')
        self.assertEqual(len(logfile_lines), 0)
        self.assertEqual(len(info['warnings']), 0)

    def test_calculate_strain_tensor(self):
        """
        Tests that calculation of strain tensor works.
        """
        cell = self.reference.cell.copy()
        cell *= 1.05
        cell[0, 0] = 0.02
        target_val = np.array([[4.49940477e-02, 2.49401921e-03, 2.49401921e-03],
                               [2.49401921e-03, 5.00089427e-02, 8.94271822e-06],
                               [2.49401921e-03, 8.94271822e-06, 5.00089427e-02]])
        strain = calculate_strain_tensor(self.reference.cell, cell)
        self.assertTrue(np.allclose(strain, target_val))


if __name__ == '__main__':
    unittest.main()
