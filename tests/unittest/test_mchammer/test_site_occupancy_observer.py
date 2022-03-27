import unittest
import inspect
import os
from random import sample

from ase.build import bulk
from ase.db import connect
from icet import ClusterSpace
from mchammer.observers.site_occupancy_observer import SiteOccupancyObserver


def _assertAlmostEqualDict(self, retval, target, places=6):
    """
    Helper function that conducts an element-wise comparison of a
    dictionary.
    """
    self.assertIsInstance(retval, type(target))
    for key, val in target.items():
        self.assertIn(key, retval)
        s = ["key: {}({})".format(key, type(key))]
        s += ["retval: {} ({})".format(retval[key], type(retval[key]))]
        s += ["target: {} ({})".format(val, type(val))]
        info = '   '.join(s)
        self.assertAlmostEqual(val, retval[key], places=places, msg=info)


unittest.TestCase.assertAlmostEqualDict = _assertAlmostEqualDict


class TestSOFObserverOneSite(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestSOFObserverOneSite, self).__init__(*args, **kwargs)

        self.chemical_symbols = ['Ag', 'Au']
        self.cutoffs = [4.0] * 3
        self.structure_prim = bulk('Ag', a=4.09)

        # 8 elements per super cell
        structure = self.structure_prim.repeat(2)
        symbols = [self.chemical_symbols[0]] * len(structure)
        symbols[2:5] = [self.chemical_symbols[1]] * 3
        structure.set_chemical_symbols(symbols)
        self.structure = structure

        self.sites = {'first4': list(range(int(len(structure) / 2))),
                      'last4': list(range(int(len(structure) / 2), len(structure), 1))}

        self.cs = ClusterSpace(self.structure_prim, self.cutoffs,
                               self.chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up observer before each test."""
        self.observer = SiteOccupancyObserver(
            self.cs, self.structure, self.sites, interval=1)

    def test_property_interval(self):
        """Tests property interval."""
        self.assertEqual(self.observer.interval, 1)

    def test_allowed_species(self):
        """Tests property interval."""
        allowed_species = {'first4': self.chemical_symbols,
                           'last4': self.chemical_symbols}
        self.assertEqual(self.observer._allowed_species,
                         allowed_species)

    def test_get_observable(self):
        """Tests observable is returned accordingly."""
        observable = {'sof_first4_Ag': 0.5,
                      'sof_first4_Au': 0.5,
                      'sof_last4_Ag': 0.75,
                      'sof_last4_Au': 0.25}
        self.assertAlmostEqualDict(self.observer.get_observable(
            structure=self.structure), observable)


class TestSOFObserverMultipleSites(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestSOFObserverMultipleSites,
              self).__init__(*args, **kwargs)

        self.chemical_symbols = [['Ag', 'Au'],
                                 ['H', 'V']]
        self.cutoffs = [5] * 2
        self.structure_prim = bulk(
            'Ag', a=4.09, crystalstructure='bcc', cubic=True).repeat([1, 1, 1])

        # 16 elements per super cell
        structure = self.structure_prim.repeat(2)
        symbols = [self.chemical_symbols[0][0],
                   self.chemical_symbols[1][0]] * int(len(structure) / 2)
        for i in range(int(len(structure) / 2), len(structure), 2):
            symbols[i] = self.chemical_symbols[0][1]
            if i >= int(3 * len(structure) / 4):
                symbols[i + 1] = self.chemical_symbols[1][1]
        structure.set_chemical_symbols(symbols)
        self.structure = structure

        self.sites = {'a': list(range(0, len(structure), 2)),
                      'b': list(range(1, len(structure), 2))}

        self.cs = ClusterSpace(self.structure_prim, self.cutoffs,
                               self.chemical_symbols)

    def setUp(self):
        """Set up observer before each test."""
        self.observer = SiteOccupancyObserver(
            self.cs, self.structure, self.sites, interval=1)

    def test_allowed_species(self):
        """Tests allowed species."""
        allowed_species = {'a': self.chemical_symbols[0],
                           'b': self.chemical_symbols[1]}
        self.assertEqual(self.observer._allowed_species,
                         allowed_species)

    def test_get_observable(self):
        """Tests observable is returned accordingly."""
        observable = {'sof_a_Ag': 0.5,
                      'sof_a_Au': 0.5,
                      'sof_b_H': 0.75,
                      'sof_b_V': 0.25}
        self.assertAlmostEqualDict(self.observer.get_observable(
            structure=self.structure), observable)


class TestSOFObserverClathrate(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestSOFObserverClathrate,
              self).__init__(*args, **kwargs)

        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(filename))
        db = connect(os.path.join(
            path, '../../structure_databases/primitive_clathrate.db'))

        self.cutoffs = [5] * 2
        self.structure_prim = db.get_atoms(id=1)
        self.chemical_symbols = [['Ba'] if s == 'Ba' else ['Ga', 'Ge'] for s in
                                 self.structure_prim.get_chemical_symbols()]

        self.sites = {'6c': [24, 25, 26, 27, 28, 29],
                      '16i': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                              21, 22, 23],
                      '24k': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                              42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]}

        self.cs = ClusterSpace(self.structure_prim, self.cutoffs,
                               self.chemical_symbols)

    def setUp(self):
        """Set up observer before each test."""
        self.observer = SiteOccupancyObserver(
            self.cs, self.structure_prim, self.sites, interval=1)

    def test_allowed_species(self):
        """Tests allowed species."""
        allowed_species = {'6c': ['Ga', 'Ge'],
                           '16i': ['Ga', 'Ge'],
                           '24k': ['Ga', 'Ge']}
        self.assertEqual(self.observer._allowed_species,
                         allowed_species)

    def test_get_observable(self):
        """Tests observable is returned accordingly."""

        def get_sofs(structure, cluster_space):
            size = len(structure) / 54
            n_cells = size**3
            cv = cluster_space.get_cluster_vector(structure)

            sofs = {}

            # 6c
            cv_6c = cv[1]
            sofs['sof_6c_Ga'] = -((6 * n_cells * cv_6c
                                   - 6 * n_cells) / (6 * 2 * size))
            sofs['sof_6c_Ge'] = 1.0 - sofs['sof_6c_Ga']

            # 16i
            cv_16i = cv[2]
            sofs['sof_16i_Ga'] = -((16 * n_cells * cv_16i
                                    - 16 * n_cells) / (16 * 2 * size))
            sofs['sof_16i_Ge'] = 1.0 - sofs['sof_16i_Ga']

            # 24k
            cv_24k = cv[3]
            sofs['sof_24k_Ga'] = -((24 * n_cells * cv_24k
                                    - 24 * n_cells) / (24 * 2 * size))
            sofs['sof_24k_Ge'] = 1.0 - sofs['sof_24k_Ga']

            return sofs

        for i in range(6, 20):
            structure = self.structure_prim.copy()
            hosts = ['Ge'] * 46
            for j in sample(range(len(hosts)), i):
                hosts[j] = 'Ga'
            structure.set_chemical_symbols(['Ba'] * 8 + hosts)
            observable = get_sofs(structure, self.cs)
            self.assertAlmostEqualDict(self.observer.get_observable(
                structure=structure), observable)


if __name__ == '__main__':
    unittest.main()
