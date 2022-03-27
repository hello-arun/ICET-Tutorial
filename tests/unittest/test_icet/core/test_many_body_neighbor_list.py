import unittest

from ase.build import bulk
from icet.core.structure import Structure
from icet.core.lattice_site import LatticeSite
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.many_body_neighbor_list import (
    ManyBodyNeighborList)


class TestManyBodyNeighborList(unittest.TestCase):
    """Container for test of the module functionality."""

    def __init__(self, *args, **kwargs):
        super(TestManyBodyNeighborList, self).__init__(*args, **kwargs)
        self.structure = bulk('Ni', 'hcp', a=3.0).repeat([2, 2, 1])
        self.cutoffs = [5.0, 5.0]
        self.position_tolerance = 1e-5

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.mbnl = ManyBodyNeighborList()
        structure = Structure.from_atoms(self.structure)
        self.neighbor_lists = get_neighbor_lists(structure, self.cutoffs, self.position_tolerance)

    def test_build(self):
        """Tests build."""
        for index in range(len(self.structure)):
            self.mbnl.build(self.neighbor_lists, index, True)

    def test_bothways_true(self):
        """
        Build the mbnl with bothways = True and assert that
        each index in the structure object have the same number
        of neighbors.
        """
        mbnl_size = len(self.mbnl.build(self.neighbor_lists, 0, True))
        for index in range(1, len(self.structure)):
            self.assertEqual(mbnl_size, len(self.mbnl.build(
                self.neighbor_lists, index, True)))

    def test_bothways_false(self):
        """
        Build the mbnl with bothways = False and assert that mbnl
        built on the first index in the structure object do not have
        the same number of neighbors as the other atoms.
        """
        mbnl_size = len(self.mbnl.build(self.neighbor_lists, 0, False))
        for index in range(1, len(self.structure)):
            self.assertNotEqual(mbnl_size, len(self.mbnl.build(
                self.neighbor_lists, index, False)))

    def test_singlets(self):
        """
        Tests that every singlet lattice site is listed
        in the many-body neighbor list.
        """
        for index in range(len(self.structure)):
            target = tuple(([LatticeSite(index, [0., 0., 0.])], []))
            singlet = self.mbnl.build(self.neighbor_lists, index, False)[0]
            self.assertEqual(singlet, target)

    def test_pairs(self):
        """
        Tests that many-body_neighbor list includes all the pairs returned
        by  neighbor_list for a specific lattice site.
        """
        index = 0
        nl_neighbors = self.neighbor_lists[0][0]
        target = tuple(([LatticeSite(index, [0., 0., 0.])], nl_neighbors))
        pairs = self.mbnl.build(self.neighbor_lists, index, True)[1]
        self.assertEqual(pairs, target)

    def test_higher_order_neighbors(self):
        """
        Tests higher order neighbors in many-body neighbor list for a
        specific lattice site.
        """
        index = 0
        high_order_neighbors = \
            self.mbnl.build(self.neighbor_lists, index, False)[2]

        target = ([LatticeSite(0, [0, 0, 0]), LatticeSite(0, [0, 0, 1])],
                  [LatticeSite(1, [0, 0, 0]),
                   LatticeSite(3, [0, -1, 0]),
                   LatticeSite(5, [-1, -1, 0]),
                   LatticeSite(5, [-1, 0, 0]),
                   LatticeSite(5, [0, 0, 0]),
                   LatticeSite(7, [-1, -1, 0])])

        self.assertEqual(target, high_order_neighbors)

    def test_calculate_intersections(self):
        """Tests intersection between two list of neighbors."""
        lattice_sites = []
        lattice_sites.append(LatticeSite(0, [0, 0, 0]))
        lattice_sites.append(LatticeSite(0, [1, 0, 0]))
        lattice_sites.append(LatticeSite(1, [0, 0, 0]))
        lattice_sites.append(LatticeSite(3, [0, 0, 0]))

        lattice_sites2 = []
        lattice_sites2.append(LatticeSite(0, [0, 0, 0]))
        lattice_sites2.append(LatticeSite(0, [1, 0, 0]))

        intersection = self.mbnl.calculate_intersection(
            lattice_sites, lattice_sites2)

        self.assertEqual(sorted(intersection), [LatticeSite(
            0, [0, 0, 0]), LatticeSite(0, [1, 0, 0])])

    def test_mbnl_non_pbc(self):
        """Tests many-body neighbor list for non-pbc structure."""
        structure = self.structure.copy()
        structure.set_pbc([False])
        neighbor_lists = get_neighbor_lists(
            Structure.from_atoms(structure), self.cutoffs, self.position_tolerance)

        mbnl = ManyBodyNeighborList()

        target = [([LatticeSite(0, [0, 0, 0])], []),
                  ([LatticeSite(0, [0, 0, 0])],
                   [LatticeSite(1, [0, 0, 0]),
                    LatticeSite(2, [0, 0, 0]),
                    LatticeSite(4, [0, 0, 0]),
                    LatticeSite(5, [0, 0, 0]),
                    LatticeSite(6, [0, 0, 0])]),
                  ([LatticeSite(0, [0, 0, 0]), LatticeSite(1, [0, 0, 0])],
                   [LatticeSite(2, [0, 0, 0]), LatticeSite(4, [0, 0, 0]),
                    LatticeSite(5, [0, 0, 0]), LatticeSite(6, [0, 0, 0])]),
                  ([LatticeSite(0, [0, 0, 0]), LatticeSite(2, [0, 0, 0])],
                   [LatticeSite(6, [0, 0, 0])]),
                  ([LatticeSite(0, [0, 0, 0]), LatticeSite(4, [0, 0, 0])],
                   [LatticeSite(5, [0, 0, 0]), LatticeSite(6, [0, 0, 0])]),
                  ([LatticeSite(0, [0, 0, 0]), LatticeSite(5, [0, 0, 0])],
                   [LatticeSite(6, [0, 0, 0])])]

        neighbors_non_pbc = mbnl.build(neighbor_lists, 0, False)

        for k, latt_neighbors in enumerate(neighbors_non_pbc):
            self.assertEqual(target[k], latt_neighbors)

    def test_mbnl_cubic_non_pbc(self):
        """
        Tests that corners sites in a large cubic cell have
        only three neighbors in many-body neighbor list.
        """
        structure = bulk('Al', 'sc', a=4.0).repeat(4)
        structure.set_pbc(False)

        neighbor_lists = get_neighbor_lists(
            Structure.from_atoms(structure), self.cutoffs, self.position_tolerance)

        mbnl = ManyBodyNeighborList()
        # atomic indices located at the corner of structure
        corner_sites = [0, 3, 12, 15, 48, 51, 60, 63]
        for index in corner_sites:
            lattice_neighbor = mbnl.build(neighbor_lists,
                                          index, True)
            # check pairs
            self.assertEqual(len(lattice_neighbor[1][1]), 3)
            # not neighbors besides above pairs
            with self.assertRaises(IndexError):
                lattice_neighbor[2]


if __name__ == '__main__':
    unittest.main()
