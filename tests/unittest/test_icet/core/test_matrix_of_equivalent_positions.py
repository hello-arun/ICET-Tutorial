import unittest
import numpy as np
import spglib

from ase.build import bulk

from icet.core.structure import Structure
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.matrix_of_equivalent_positions import (
    MatrixOfEquivalentPositions, matrix_of_equivalent_positions_from_structure)
from icet.core.matrix_of_equivalent_positions import (
    _get_lattice_site_matrix_of_equivalent_positions as
    get_lattice_site_matrix_of_equivalent_positions)
from icet.core.matrix_of_equivalent_positions import (
    _fractional_to_cartesian as fractional_to_cartesian)
from icet.core.matrix_of_equivalent_positions import (
    _prune_matrix_of_equivalent_positions as prune_matrix_of_equivalent_positions)
from icet.tools.geometry import (
    ase_atoms_to_spglib_cell,
    get_primitive_structure,
    get_fractional_positions_from_neighbor_list)


class TestMatrixOfEquivalentPositions(unittest.TestCase):
    """Container for test of the module functionality."""

    def __init__(self, *args, **kwargs):
        super(TestMatrixOfEquivalentPositions, self).__init__(*args, **kwargs)

        self.position_tolerance = 1e-6
        self.symprec = 1e-6
        self.fractional_position_tolerance = 1e-7
        self.structure = bulk('Ni', 'hcp', a=3.0).repeat([2, 2, 1])
        self.cutoff = 5.0
        self.structure_prim = get_primitive_structure(self.structure)
        icet_structure_prim = Structure.from_atoms(self.structure_prim)
        neighbor_list = get_neighbor_lists(self.structure_prim,
                                           cutoffs=[self.cutoff],
                                           position_tolerance=self.position_tolerance)[0]
        self.frac_positions = get_fractional_positions_from_neighbor_list(
            icet_structure_prim, neighbor_list)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        symmetry = spglib.get_symmetry(ase_atoms_to_spglib_cell(self.structure_prim))
        self.translations = symmetry['translations']
        self.rotations = symmetry['rotations']

        self.pm = MatrixOfEquivalentPositions(self.translations, self.rotations)
        self.pm.build(self.frac_positions)

    def test_init(self):
        """Test initializer."""
        self.assertIsInstance(self.pm, MatrixOfEquivalentPositions)

    def test_init_with_nonmatching_symmetries(self):
        """Test that exception is raised when symmetries do not match."""
        with self.assertRaises(ValueError) as context:
            MatrixOfEquivalentPositions(self.translations, self.rotations[:-1])
        self.assertIn('The number of translations', str(context.exception))

    def test_dimension_matrix_of_equivalent_positions(self):
        """
        Tests dimensions of permutation matrix. Number of rows should
        be equal to the number of symmetry operations while number of columns
        must correpond to the total number of fractional positions.
        """
        pm_frac = self.pm.get_equivalent_positions()
        for row in pm_frac:
            self.assertEqual(len(row), len(self.rotations))
        self.assertEqual(len(pm_frac), len(self.frac_positions))

    def test_get_equivalent_positions(self):
        """
        Tests that first row and first column of permutation matrix match
        the target lists.
        """
        pm_frac = self.pm.get_equivalent_positions()

        target_row = [[0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0]]
        retval_row = [pos.tolist() for pos in pm_frac[0]]
        np.testing.assert_array_almost_equal(
            np.array(sorted(target_row)), np.array(sorted(retval_row)), decimal=5)

        target_col = [[0.0, 0.0, 0.0],
                      [-1.0, -1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [-0.6666667, -1.3333333, -0.5],
                      [-0.6666667, -1.3333333, 0.5],
                      [-0.6666667, -0.3333333, -0.5],
                      [-0.6666667, -0.3333333, 0.5],
                      [-0.6666667, 0.6666667, -0.5],
                      [-0.6666667, 0.6666667, 0.5],
                      [0.3333333, -0.3333333, -0.5],
                      [0.3333333, -0.3333333, 0.5],
                      [0.3333333, 0.6666667, -0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [1.3333333, 0.6666667, -0.5],
                      [1.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [-1.0, 0.0, 0.0],
                      [-1.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 0.0, 1.0],
                      [1.0, 1.0, 0.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 2.0, 0.0],
                      [1.0, 2.0, 1.0],
                      [-0.6666667, -0.3333333, 0.5],
                      [-0.6666667, 0.6666667, 0.5],
                      [0.3333333, -0.3333333, 0.5],
                      [0.3333333, 0.6666667, -0.5],
                      [0.3333333, 0.6666667, 1.5],
                      [0.3333333, 1.6666667, 0.5],
                      [1.3333333, 0.6666667, 0.5],
                      [1.3333333, 1.6666667, 0.5]]

        retval_col = [row[0].tolist() for row in pm_frac]
        np.testing.assert_array_almost_equal(
            np.array(sorted(target_col)), np.array(sorted(retval_col)), decimal=5)

    def test_matrix_of_equivalent_positions_from_structure(self):
        """Tests permutation matrix from structure functionality."""
        pm, _, _ = matrix_of_equivalent_positions_from_structure(self.structure, self.cutoff,
                                                                 self.position_tolerance,
                                                                 self.symprec)

        matrix = pm.get_equivalent_positions()
        matrix2 = self.pm.get_equivalent_positions()

        for row, row2 in zip(matrix, matrix2):
            self.assertEqual(len(row), len(row2))
            for element, element2 in zip(row, row2):
                self.assertEqual(element.tolist(), element2.tolist())

        pm_prim, _, _ = \
            matrix_of_equivalent_positions_from_structure(self.structure_prim, self.cutoff,
                                                          self.position_tolerance, self.symprec,
                                                          find_primitive=False)

        matrix_prim = pm_prim.get_equivalent_positions()

        for row, row2 in zip(matrix, matrix_prim):
            self.assertEqual(len(row), len(row2))
            for element, element2 in zip(row, row2):
                self.assertEqual(element.tolist(), element2.tolist())

    def test_fractional_to_cartesian(self):
        """
        Tests fractional coordinates are converted into cartesians coordinates.
        """
        target = [[0.0, 0.0, 0.0],
                  [-0.0, 1.73, 2.45],
                  [-0.0, 1.73, 2.45],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [-0.0, 1.73, 2.45],
                  [-0.0, 1.73, 2.45],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [-0.0, 1.73, 2.45],
                  [-0.0, 1.73, 2.45],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [-0.0, 1.73, 2.45],
                  [-0.0, 1.73, 2.45],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [-0.0, 1.73, 2.45],
                  [-0.0, 1.73, 2.45],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [-0.0, 1.73, 2.45],
                  [-0.0, 1.73, 2.45],
                  [0.0, 0.0, 0.0]]

        fractional_pos = self.pm.get_equivalent_positions()[0]
        cartesian_pos = fractional_to_cartesian(
            fractional_pos, self.structure_prim.cell)
        retval = np.around(cartesian_pos, decimals=2).tolist()
        self.assertListEqual(retval, target)

    def test_lattice_site_matrix_of_equivalent_positions(self):
        """
        Tests lattice sites in permutation matrix by asserting the distances
        between r_ik and r_jk sites in the same column.
        """
        # TODO: Some part of the implementation cannot be covered as test fails
        # for non-pbc structures.
        structure = bulk('Al').repeat(2)
        cutoff = 4.2
        pm, prim_structure, _ = \
            matrix_of_equivalent_positions_from_structure(structure, cutoff,
                                                          self.position_tolerance,
                                                          self.symprec)
        pm_lattice_site = get_lattice_site_matrix_of_equivalent_positions(
            prim_structure, pm, self.fractional_position_tolerance)
        for i in range(len(pm_lattice_site)):
            for j in range(i + 1, len(pm_lattice_site)):
                dist_last = -1
                for k in range(len(pm_lattice_site[i])):
                    site_1 = pm_lattice_site[i][k]
                    site_2 = pm_lattice_site[j][k]
                    pos1 = self.structure[site_1.index].position +\
                        np.dot(site_1.unitcell_offset, structure.cell)
                    pos2 = self.structure[site_2.index].position +\
                        np.dot(site_2.unitcell_offset, structure.cell)
                    dist_first = np.linalg.norm(pos1 - pos2)
                    if dist_last != -1:
                        self.assertAlmostEqual(dist_first, dist_last, places=8)
                    dist_last = dist_first

    def test_prune_matrix_of_equivalent_positions(self):
        """
        Tests that first column of pruned permutation matrix
        containes unique elements.
        """
        pm, prim_structure, _ = matrix_of_equivalent_positions_from_structure(
            self.structure, self.cutoff, self.position_tolerance, self.symprec)

        pm_lattice_site = get_lattice_site_matrix_of_equivalent_positions(
            prim_structure, pm, self.fractional_position_tolerance)

        pruned_matrix = prune_matrix_of_equivalent_positions(pm_lattice_site)
        first_col = []
        for row in pruned_matrix:
            first_col.append(row[0])
        for i, site_i in enumerate(first_col):
            for j, site_j in enumerate(first_col):
                if i <= j:
                    continue
                else:
                    self.assertNotEqual(site_i, site_j)


if __name__ == '__main__':
    unittest.main()
