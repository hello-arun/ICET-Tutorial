import unittest
from ase.build import bulk, make_supercell
import numpy as np

from icet.core.orbit_list import OrbitList, Structure
from icet.core.lattice_site import LatticeSite
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator


class TestLocalOrbitListGenerator(unittest.TestCase):
    """Container for test of class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestLocalOrbitListGenerator, self).__init__(*args, **kwargs)
        self.symprec = 1e-5
        self.position_tolerance = 1e-5
        self.fractional_position_tolerance = 1e-6

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiate class for each test case."""
        prim_structure = bulk('Al')
        cutoffs = [4.2, 4.2]
        chemical_symbols = [['Al', 'Zn']]
        self.orbit_list = OrbitList(
            prim_structure, cutoffs, chemical_symbols,
            symprec=self.symprec, position_tolerance=self.position_tolerance,
            fractional_position_tolerance=self.fractional_position_tolerance)
        self.primitive = self.orbit_list.get_structure()
        super_structure = make_supercell(prim_structure, [[2, 0, 1000],
                                                          [0, 2, 0],
                                                          [0, 0, 2]])
        self.supercell = Structure.from_atoms(super_structure)

        self.lolg = LocalOrbitListGenerator(
            self.orbit_list, self.supercell,
            fractional_position_tolerance=self.fractional_position_tolerance)

    def test_generate_local_orbit_list(self):
        """
        Tests that function generates an orbit list from
        an index of a specific offset of the primitive structure.
        """
        for offset in self.lolg._get_unique_primcell_offsets():
            local_orbit_list = self.lolg.generate_local_orbit_list(offset, False)
            for orbit_prim, orbit_super in zip(self.orbit_list.orbits,
                                               local_orbit_list.orbits):
                for site_p, site_s in zip(orbit_prim.representative_cluster.lattice_sites,
                                          orbit_super.representative_cluster.lattice_sites):
                    site_p.unitcell_offset += offset
                    pos_super = self.supercell.get_position(site_s)
                    pos_prim = self.primitive.get_position(site_p)
                    self.assertTrue(np.all(np.isclose(pos_super, pos_prim)))

                    # Since we have modified the lattice site, we need to restore it
                    site_p.unitcell_offset -= offset

    def test_generating_full_orbit_list_with_primitive(self):
        """
        Tests creating a full orbit list using the primitive as the supercell.
        """
        prim_structure = bulk('Al')
        prim = Structure.from_atoms(prim_structure)
        lolg = LocalOrbitListGenerator(
            self.orbit_list, prim,
            fractional_position_tolerance=self.fractional_position_tolerance)
        lolg.generate_full_orbit_list()

    def test_generate_full_orbit_list(self):
        """
        Tests that equivalent sites of all local orbit lists are listed
        as equivalent sites in the full orbit list.
        """
        print('nu')
        fol = self.lolg.generate_full_orbit_list()
        for offset in self.lolg._get_unique_primcell_offsets():
            print(offset)
            lol = self.lolg.generate_local_orbit_list(offset)
            for orbit, orbit_ in zip(lol.orbits, fol.orbits):
                for cluster in orbit.clusters:
                    self.assertIn(cluster.lattice_sites,
                                  [cluster.lattice_sites for cluster in orbit_.clusters])

    def test_unique_offset_count(self):
        """
        Tests number of unique offsets corresponds to the number of atoms
        in the supercell given that there is one atom in the primitive cell.
        """
        self.assertEqual(self.lolg.get_number_of_unique_offsets(),
                         len(self.supercell))

    def test_unique_primcell_offsets(self):
        """
        Tests primitive offsets are unique and take to positions that
        match atoms positions in the supercell.
        """
        unique_offsets = self.lolg._get_unique_primcell_offsets()
        super_pos = self.supercell.positions

        for k, offset in enumerate(unique_offsets):
            pos_prim = self.primitive.get_position(LatticeSite(0, offset))
            self.assertTrue(
                np.any(np.isclose(pos_prim, pos) for pos in super_pos))
            for i in range(k + 1, len(unique_offsets)):
                self.assertFalse(np.all(np.isclose(offset, unique_offsets[i])))


class TestLocalOrbitListGeneratorHCP(unittest.TestCase):
    """
    Container for test of class functionality for hcp structure,
    which contains two atoms per unitcell.
    """

    def __init__(self, *args, **kwargs):
        super(TestLocalOrbitListGeneratorHCP, self).__init__(*args, **kwargs)
        prim_structure = bulk('Ni', 'hcp', a=4.0)
        cutoffs = [4.2, 4.2]
        chemical_symbols = [['Al', 'Zn'], ['Al', 'Zn']]
        self.symprec = 1e-5
        self.position_tolerance = 1e-5
        self.fractional_position_tolerance = 1e-6
        self.orbit_list = OrbitList(
            prim_structure, cutoffs, chemical_symbols,
            symprec=self.symprec, position_tolerance=self.position_tolerance,
            fractional_position_tolerance=self.fractional_position_tolerance)
        self.primitive = self.orbit_list.get_structure()
        super_structure = make_supercell(prim_structure, [[2, 0, 1000],
                                                          [0, 2, 0],
                                                          [0, 0, 2]])
        self.supercell = Structure.from_atoms(super_structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiate class for each test case."""
        self.lolg = LocalOrbitListGenerator(
            self.orbit_list, self.supercell, self.fractional_position_tolerance)

    def test_generate_local_orbit_list(self):
        """
        Tests that function generates an orbit list for the given
        offset of the primitive structure.
        """
        for offset in self.lolg._get_unique_primcell_offsets():
            local_orbit_list = self.lolg.generate_local_orbit_list(offset)
            for orbit_prim, orbit_super in zip(self.orbit_list.orbits,
                                               local_orbit_list.orbits):
                for site_p, site_s in zip(orbit_prim.representative_cluster.lattice_sites,
                                          orbit_super.representative_cluster.lattice_sites):
                    site_p.unitcell_offset += offset
                    pos_super = self.supercell.get_position(site_s)
                    pos_prim = self.primitive.get_position(site_p)
                    self.assertTrue(np.allclose(pos_super, pos_prim))

                    # Since we have modified the lattice site, we need to restore it
                    site_p.unitcell_offset -= offset

    def test_unique_offset_count(self):
        """
        Tests number of unique offsets corresponds to half of the total number
        of atoms in the supercell given that there is two atoms per unitcell.
        """
        self.assertEqual(self.lolg.get_number_of_unique_offsets(),
                         len(self.supercell) / 2)

    def test_unique_primcell_offsets(self):
        """
        Tests primitive offsets are unique and take to positions that
        match atoms positions in the supercell.
        """
        unique_offsets = self.lolg._get_unique_primcell_offsets()
        super_pos = self.supercell.positions

        for k, offset in enumerate(unique_offsets):
            pos_prim = self.primitive.get_position(LatticeSite(0, offset))
            self.assertTrue(
                np.any(np.isclose(pos_prim, pos) for pos in super_pos))
            for i in range(k + 1, len(unique_offsets)):
                self.assertFalse(np.all(np.isclose(offset, unique_offsets[i])))


if __name__ == '__main__':
    unittest.main()
