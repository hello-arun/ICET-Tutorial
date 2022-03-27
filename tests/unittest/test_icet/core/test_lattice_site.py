import unittest

from icet.core.lattice_site import LatticeSite
import numpy as np
import itertools


class TestLatticeSite(unittest.TestCase):
    """Container for test of class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestLatticeSite, self).__init__(*args, **kwargs)
        self.indices = [i for i in range(8)]
        self.unitcell_offsets = []
        cartesian_product_lists = [[0., 1.], [0., 1.], [0., 1.]]
        for element in itertools.product(*cartesian_product_lists):
            self.unitcell_offsets.append(list(element))

    def setUp(self):
        """Setup before each test."""
        self.lattice_sites = []
        for index, unitcell_offset in zip(self.indices, self.unitcell_offsets):
            lattice_site = LatticeSite(index, unitcell_offset)
            self.lattice_sites.append(lattice_site)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_index_property(self):
        """Tests index property."""
        self.assertIsInstance(self.lattice_sites[0].index, int)
        self.assertEqual(self.lattice_sites[0].index, 0)

    def test_offset_property(self):
        """Tests unitcell_offset property."""
        self.assertIsInstance(
            self.lattice_sites[0].unitcell_offset, type(np.array([0])))
        self.assertIsInstance(
            self.lattice_sites[0].unitcell_offset,
            type(self.lattice_sites[0].unitcell_offset))
        self.assertEqual(
            list(self.lattice_sites[0].unitcell_offset), [0., 0., 0.])

    def test_eq(self):
        """Tests eq operator."""
        index = 152453453
        unitcell_offset = [-234234., 32423423., 235567567.]

        lattice_site = LatticeSite(index, unitcell_offset)
        lattice_site_other = LatticeSite(index, unitcell_offset)
        self.assertEqual(lattice_site, lattice_site_other)
        self.assertNotEqual(lattice_site, self.lattice_sites[0])
        self.assertNotEqual(self.lattice_sites[1], self.lattice_sites[0])

    def test_less_than(self):
        """Tests less than operator."""
        self.assertLess(self.lattice_sites[0], self.lattice_sites[1])

    def test_add_operator(self):
        """Tests adding operator."""
        lattice_site = LatticeSite(0, [0, 0, 0])
        lattice_site2 = LatticeSite(0, [-1, -1, 3])
        lattice_site.unitcell_offset = \
            lattice_site.unitcell_offset + [-1, -1, 3]
        self.assertEqual(lattice_site, lattice_site2)

    def test_substraction_operator(self):
        """Tests substraction operator."""
        lattice_site = LatticeSite(0, [-1, -1, 3])
        lattice_site.unitcell_offset = \
            lattice_site.unitcell_offset - [-1, -1, 3]
        self.assertEqual(lattice_site, LatticeSite(0, [0, 0, 0]))

    def test_add_assigment_operator(self):
        """Tests adding and assignment operator."""
        lattice_site = LatticeSite(0, [0, 0, 0])
        lattice_site2 = LatticeSite(0, [-1, -1, 3])
        lattice_site2.unitcell_offset += [1, 1, -3]
        self.assertEqual(lattice_site, lattice_site2)

    def test_str(self):
        """Tests the string representation of LatticeSite."""
        target = '0 : [0 0 0]'
        retval = str(self.lattice_sites[0])
        self.assertEqual(target, retval)

    def test_hash(self):
        """Tests that lattice site is hashable (check)."""
        index = 152453453
        unitcell_offset = [-234234., 32423423., 235567567.]

        lattice_site = LatticeSite(index, unitcell_offset)
        lattice_site_other = LatticeSite(index, unitcell_offset)
        self.assertEqual(lattice_site.__hash__(),
                         lattice_site_other.__hash__())


if __name__ == '__main__':
    unittest.main()
