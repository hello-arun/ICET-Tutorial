import unittest
from mchammer.configuration_manager import ConfigurationManager
from ase.build import bulk
from icet import ClusterSpace
from mchammer.configuration_manager import SwapNotPossibleError


class TestConfigurationManager(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestConfigurationManager, self).__init__(*args, **kwargs)
        self.structure = bulk('Al').repeat([2, 1, 1])
        self.structure[1].symbol = 'Ag'
        self.structure = self.structure.repeat(3)
        cs = ClusterSpace(self.structure, cutoffs=[0], chemical_symbols=['Ag', 'Al'])
        self.sublattices = cs.get_sublattices(self.structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        self.cm = ConfigurationManager(self.structure, self.sublattices)

    def test_type(self):
        """Tests cm type."""
        self.assertIsInstance(self.cm, ConfigurationManager)

    def test_property_structure(self):
        """Tests structure property."""
        self.assertEqual(self.structure, self.cm.structure)

    def test_property_occupations(self):
        """
        Tests that the occupation property returns expected result
        and that it cannot be modified.
        """

        self.assertListEqual(list(self.cm.occupations),
                             list(self.structure.numbers))

        # Tests that the property can not be set.
        with self.assertRaises(AttributeError) as context:
            self.cm.occupations = []
        self.assertTrue("can't set attribute" in str(context.exception))

        # Tests that the property can't be set by modifying the
        # returned occupations
        occupations = self.cm.occupations
        occupations[0] = occupations[0] + 1
        self.assertNotEqual(list(occupations), list(self.cm.occupations))

    def test_property_sublattices(self):
        """
        Tests that the occupation_constraints property returns expected result
        and that it cannot be modified.
        """

        self.assertEqual(self.cm.sublattices,
                         self.sublattices)

        # Tests that the property can not be set.
        with self.assertRaises(AttributeError) as context:
            self.cm.sublattices = []
        self.assertTrue("can't set attribute" in str(context.exception))

    def test_get_occupations_on_sublattice(self):
        """Tests get_occupations_on_sublattice function."""

        # get all occupations
        indices = self.cm.sublattices[0].indices
        target = list(self.cm.occupations[indices])
        self.assertListEqual(target, self.cm.get_occupations_on_sublattice(0))

    def test_is_swap_possible(self):
        """Tests is_swap_possible function."""

        # swaps are possible
        for i, sl in enumerate(self.cm.sublattices):
            self.assertTrue(self.cm.is_swap_possible(i))

        # setup system with inactive sublattice
        prim = bulk('Al').repeat([2, 1, 1])
        chemical_symbols = [['Al'], ['Ag', 'Al', 'Au']]
        cs = ClusterSpace(prim, cutoffs=[0], chemical_symbols=chemical_symbols)

        supercell = prim.repeat(2)
        supercell[1].symbol = 'Ag'
        supercell[3].symbol = 'Au'
        sublattices = cs.get_sublattices(supercell)
        cm = ConfigurationManager(supercell, sublattices)

        # check both sublattices
        self.assertTrue(cm.is_swap_possible(0))
        self.assertFalse(cm.is_swap_possible(1))

        # check both sublattices when specifying allowed species
        allowed_species = [13, 47]
        self.assertTrue(cm.is_swap_possible(0,
                                            allowed_species=allowed_species))
        self.assertFalse(cm.is_swap_possible(1,
                                             allowed_species=allowed_species))

    def test_get_swapped_state(self):
        """Tests the getting swap indices method."""

        for _ in range(1000):
            indices, elements = self.cm.get_swapped_state(0)
            index1 = indices[0]
            index2 = indices[1]
            self.assertNotEqual(
                self.cm.occupations[index1], self.cm.occupations[index2])
            self.assertNotEqual(
                elements[0], elements[1])
            self.assertEqual(self.cm.occupations[index1], elements[1])
            self.assertEqual(self.cm.occupations[index2], elements[0])

        # set everything to Al and see that swap is not possible
        indices = [i for i in range(len(self.structure))]
        elements = [13] * len(self.structure)
        self.cm.update_occupations(indices, elements)

        with self.assertRaises(SwapNotPossibleError) as context:
            indices, elements = self.cm.get_swapped_state(0)
        self.assertTrue("Cannot swap on sublattice" in str(context.exception))
        self.assertTrue("since it is full of" in str(context.exception))

        # setup a ternary system
        prim = bulk('Al').repeat([3, 1, 1])
        chemical_symbols = ['Ag', 'Al', 'Au']
        cs = ClusterSpace(prim, cutoffs=[0], chemical_symbols=chemical_symbols)

        for i, symbol in enumerate(chemical_symbols):
            prim[i].symbol = symbol
        supercell = prim.repeat(2)
        sublattices = cs.get_sublattices(supercell)
        cm = ConfigurationManager(supercell, sublattices)

        allowed_species = [13, 47]
        for _ in range(1000):
            indices, elements = cm.get_swapped_state(
                0, allowed_species=allowed_species)
            index1 = indices[0]
            index2 = indices[1]
            self.assertNotEqual(
                cm.occupations[index1], cm.occupations[index2])
            self.assertNotEqual(
                elements[0], elements[1])
            self.assertEqual(cm.occupations[index1], elements[1])
            self.assertEqual(cm.occupations[index2], elements[0])

        # set everything to Al and see that swap is not possible
        indices = [i for i in range(len(supercell))]
        elements = [13] * len(supercell)
        cm.update_occupations(indices, elements)

        with self.assertRaises(SwapNotPossibleError) as context:
            indices, elements = cm.get_swapped_state(
                0, allowed_species=allowed_species)
        self.assertTrue("Cannot swap on sublattice" in str(context.exception))
        self.assertTrue("since it is full of" in str(context.exception))

    def test_get_flip_index(self):
        """Tests the getting flip indices method."""

        for _ in range(1000):
            index, element = self.cm.get_flip_state(0)
            self.assertNotEqual(self.cm.occupations[index], element)

        # setup a ternary system
        prim = bulk('Al').repeat([3, 1, 1])
        chemical_symbols = ['Ag', 'Al', 'Au']
        cs = ClusterSpace(prim, cutoffs=[0], chemical_symbols=chemical_symbols)

        for i, symbol in enumerate(chemical_symbols):
            prim[i].symbol = symbol
        supercell = prim.repeat(2)
        sublattices = cs.get_sublattices(supercell)
        cm = ConfigurationManager(supercell, sublattices)

        allowed_species = [13, 47]
        for _ in range(1000):
            index, element = cm.get_flip_state(
                0, allowed_species=allowed_species)
            self.assertNotEqual(cm.occupations[index], element)

    def test_update_occupations(self):
        """Tests the update occupation method."""
        structure_cpy = self.structure.copy()
        indices = [0, 2, 3, 5, 7, 8]
        elements = [13, 13, 47, 47, 13, 47]

        self.assertNotEqual(list(self.cm.occupations[indices]), list(elements))
        self.assertTrue(self._is_sites_by_species_dict_correct(self.cm))
        self.cm.update_occupations(indices, elements)
        self.assertEqual(list(self.cm.occupations[indices]), elements)
        self.assertTrue(self._is_sites_by_species_dict_correct(self.cm))

        # test input structure remains unchanged
        self.assertEqual(self.structure, structure_cpy)

        # test that correct exceptions are raised
        with self.assertRaises(ValueError) as context:
            self.cm.update_occupations([-1], [1])
        self.assertTrue('Site -1 is not a valid site index' in str(context.exception))
        with self.assertRaises(ValueError) as context:
            self.cm.update_occupations([0], [-1])
        self.assertTrue('Invalid new species' in str(context.exception))

    def test_sites_by_species(self):
        """Tests the element occupation dict."""

        # Initially consistent
        self.assertTrue(self._is_sites_by_species_dict_correct(self.cm))

        # Tests that changing occupations manually is wrong
        element = self.cm._occupations[0]
        self.cm._occupations[0] = 200
        self.assertFalse(self._is_sites_by_species_dict_correct(self.cm))

        # Fix error
        self.cm._occupations[0] = element
        self.assertTrue(self._is_sites_by_species_dict_correct(self.cm))

        # Set everything to Al
        indices = [i for i in range(len(self.structure))]
        elements = [13] * len(self.structure)
        self.cm.update_occupations(indices, elements)
        self.assertTrue(self._is_sites_by_species_dict_correct(self.cm))

        # Set everything to Ag
        indices = [i for i in range(len(self.structure))]
        elements = [47] * len(self.structure)
        self.cm.update_occupations(indices, elements)
        self.assertTrue(self._is_sites_by_species_dict_correct(self.cm))

        # Set everything to Al-Ag-Al-Ag ...
        indices = [i for i in range(len(self.structure))]
        elements = [13, 47] * (len(self.structure) // 2)
        self.cm.update_occupations(indices, elements)
        self.assertTrue(self._is_sites_by_species_dict_correct(self.cm))

    def _is_sites_by_species_dict_correct(self, configuration_manager):
        """
        Checks that the internal Element -> site dict is consistent
        with the occupation list.

        Parameters
        ----------
        configuration_manager : ConfigurationManager

        Return : bool
        True if the dict is correct
        False if the dict is inconsistent with self.occupations
        """

        for element_dict in configuration_manager._sites_by_species:
            for element in element_dict.keys():
                for index in element_dict[element]:
                    if element != configuration_manager.occupations[index]:
                        return False
        return True


if __name__ == '__main__':
    unittest.main()
