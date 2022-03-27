import unittest

from ase.build import bulk
from icet.core.sublattices import Sublattices, Sublattice
from icet import ClusterSpace


class TestSublattice(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestSublattice,
              self).__init__(*args, **kwargs)

        self.chemical_symbols = ['Al', 'Ge', 'Si']
        self.indices = [1, 2, 3, 4, 5, 6, 7]
        self.symbol = 'A'

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up sublattice before each test."""
        self.sublattice = Sublattice(
            chemical_symbols=self.chemical_symbols, indices=self.indices, symbol=self.symbol)

    def test_indices(self):
        """Tests indices property."""

        self.assertEqual(self.sublattice.indices, [1, 2, 3, 4, 5, 6, 7])

        # Test that indices are copied
        indices = self.sublattice.indices
        indices[0] = -99
        indices.append(37)
        self.assertEqual(self.sublattice.indices, [1, 2, 3, 4, 5, 6, 7])

    def test_chemical_symbols(self):
        """Tests chemical symbols property."""
        self.assertEqual(self.sublattice.chemical_symbols, ['Al', 'Ge', 'Si'])

        # Test that symbols are copied
        symbols = self.sublattice.chemical_symbols
        symbols[0] = 'H'
        symbols.append('Pt')
        self.assertEqual(self.sublattice.chemical_symbols, ['Al', 'Ge', 'Si'])

    def test_atomic_numbers(self):
        """Tests chemical symbols property."""
        self.assertEqual(self.sublattice.atomic_numbers, [13, 32, 14])

        # Test that symbols are copied
        numbers = self.sublattice.atomic_numbers
        numbers[0] = 1
        numbers.append(99)
        self.assertEqual(self.sublattice.atomic_numbers, [13, 32, 14])

    def test_symbol(self):
        """Tests symbol property."""
        self.assertEqual('A', self.sublattice.symbol)


class TestSublattices(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestSublattices,
              self).__init__(*args, **kwargs)
        self.prim = bulk('Au').repeat([2, 1, 1])
        self.prim[1].symbol = 'H'
        self.allowed_species = [('Pd', 'Au'), ('H', 'V')]
        self.fractional_position_tolerance = 1e-7
        self.supercell = self.prim.repeat(3)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up sublattices before each test."""
        self.sublattices = Sublattices(
            allowed_species=self.allowed_species,
            primitive_structure=self.prim,
            structure=self.supercell,
            fractional_position_tolerance=self.fractional_position_tolerance)

    def test_sublattice_ordering(self):
        """Tests ordering of sublattices."""
        allowed_species = [('H'), ['He'], ('Pd', 'Au'), ('H', 'V')]
        prim = self.prim.repeat([2, 1, 1])
        supercell = prim.repeat(3)
        sublattices = Sublattices(allowed_species=allowed_species,
                                  primitive_structure=prim,
                                  structure=supercell,
                                  fractional_position_tolerance=self.fractional_position_tolerance)

        ret = sublattices.allowed_species
        target = [('Au', 'Pd'), ('H', 'V'), ('H',), ('He',)]
        self.assertEqual(ret, target)

    def test_allowed_species(self):
        """Tests the allowed species property."""
        # Note that the Au, Pd order has changed due to lexicographically sorted symbols
        self.assertEqual(self.sublattices.allowed_species,
                         [('Au', 'Pd'), ('H', 'V')])

    def test_get_sublattice_sites(self):
        """Tests the get sublattice sites method."""

        self.assertEqual(self.sublattices.get_sublattice_sites(
            index=0), self.sublattices[0].indices)

        self.assertEqual(self.sublattices[0].indices, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
                                                       24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
                                                       46, 48, 50, 52])

        self.assertEqual(self.sublattices[1].indices, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
                                                       23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
                                                       43, 45, 47, 49, 51, 53])

    def test_allowed_numbers_on_site(self):
        """Tests the get allowed numbers on site method."""

        for atom in self.supercell:
            sublattice_index = self.sublattices.get_sublattice_index_from_site_index(atom.index)
            allowed_numbers = self.sublattices.get_allowed_numbers_on_site(atom.index)
            self.assertEqual(allowed_numbers, self.sublattices[sublattice_index].atomic_numbers)

    def test_allowed_symbols_on_site(self):
        """Tests the get allowed numbers on site method."""

        for atom in self.supercell:
            sublattice_index = self.sublattices.get_sublattice_index_from_site_index(atom.index)
            allowed_symbols = self.sublattices.get_allowed_symbols_on_site(atom.index)
            self.assertEqual(allowed_symbols, self.sublattices[sublattice_index].chemical_symbols)

    def test_get_sublattice_index_from_site_index(self):
        """Tests the get sublattice index method."""

        for i in range(len(self.supercell)):
            sublattice_index = self.sublattices.get_sublattice_index_from_site_index(index=i)
            if i % 2 == 0:
                self.assertEqual(sublattice_index, 0)
            else:
                self.assertEqual(sublattice_index, 1)

    def test_active_sublattices(self):
        """Tests the active sublattices property."""
        active_sublattices = self.sublattices.active_sublattices

        symbols_ret = [sl.chemical_symbols for sl in active_sublattices]
        target = [('Au', 'Pd'), ('H', 'V')]
        self.assertEqual(symbols_ret, target)

    def test_inactive_sublattices(self):
        """Tests the active sublattices property."""
        inactive_sublattices = self.sublattices.inactive_sublattices

        symbols_ret = [sl.chemical_symbols for sl in inactive_sublattices]
        target = []
        self.assertEqual(symbols_ret, target)

        # Now create something that actually have inactive sublattices

        sublattices = Sublattices(allowed_species=[['Au', 'Pd'], ['H']],
                                  primitive_structure=self.prim,
                                  structure=self.supercell,
                                  fractional_position_tolerance=self.fractional_position_tolerance)

        inactive_sublattices = sublattices.inactive_sublattices

        symbols_ret = [sl.chemical_symbols for sl in inactive_sublattices]
        target = [('H',)]
        self.assertEqual(symbols_ret, target)

    def test_sublattice_uniqueness(self):
        """Tests that the number of sublattices are correct
        in the case of the allowed species have duplicates in them.
        """
        structure = bulk("Al").repeat(2)

        chemical_symbols = [['H']] + [['Al', 'Ge']]*(len(structure)-1)
        cs = ClusterSpace(
            structure=structure, chemical_symbols=chemical_symbols, cutoffs=[5])
        sublattices = cs.get_sublattices(structure)

        self.assertEqual(len(sublattices), 2)
        self.assertEqual(sublattices.allowed_species, [('Al', 'Ge'), ('H',)])

    def test_sublattice_assert_occupation(self):
        """Tests the assert occupation method of sublattices."""

        # Should work out of the box
        self.sublattices.assert_occupation_is_allowed(self.supercell.get_chemical_symbols())

        # This should not work...
        chemical_symbols = self.supercell.get_chemical_symbols()
        chemical_symbols[0] = 'Si'

        with self.assertRaises(ValueError) as context:
            self.sublattices.assert_occupation_is_allowed(chemical_symbols)
        self.assertTrue(
            'Occupations of structure not compatible with the sublattice' in str(context.exception))

        # wrong length of chemical symbols should not work either
        with self.assertRaises(ValueError) as context:
            self.sublattices.assert_occupation_is_allowed(['Al', 'Ge'])
        self.assertTrue('len of input chemical symbols (2)' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
