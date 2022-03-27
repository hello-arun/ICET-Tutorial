import unittest

from ase.build import bulk
from icet import ClusterSpace
from mchammer.observers import ClusterCountObserver


class TestClusterCountObserver(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterCountObserver, self).__init__(*args, **kwargs)

        self.structure = bulk('Al').repeat([2, 1, 1])
        self.structure[1].symbol = 'Ge'

        cutoffs = [3]
        subelements = ['Al', 'Ge', 'Si']
        self.cs = ClusterSpace(self.structure, cutoffs, subelements)
        self.interval = 10

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up observer before each test."""
        self.observer = ClusterCountObserver(
            cluster_space=self.cs, structure=self.structure, interval=self.interval)

    def test_property_tag(self):
        """Tests property tag."""
        self.assertEqual(self.observer.tag, "ClusterCountObserver")

    def test_property_interval(self):
        """Tests property interval."""
        self.assertEqual(self.observer.interval, self.interval)

    def test_get_observable(self):
        """Tests observable is returned accordingly."""

        prim = bulk('Au')
        structure = prim.repeat(3)
        cutoffs = [3]
        subelements = ['Au', 'Pd']
        cs = ClusterSpace(prim, cutoffs, subelements)
        observer = ClusterCountObserver(
            cluster_space=cs, structure=structure, interval=self.interval)

        structure.set_chemical_symbols(['Au'] * len(structure))
        # 1 Pd in pure Au sro
        structure[0].symbol = 'Pd'
        counts = observer.get_observable(structure)

        # In total there will be 12 Pd neighboring an Au atom
        expected_Au_Pd_count = 12
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Pd_count, actual_counts)

        # Number of Au-Au neighbors should be
        expected_Au_Au_count = 6 * len(structure) - 12
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' not in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Au_count, actual_counts)

        # Number of Pd-Pd neighbors should be
        expected_Au_Au_count = 0
        actual_counts = 0
        for count in counts.keys():
            if 'Pd' in count and '1' in count and 'Au' not in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Au_count, actual_counts)

        # 1 Au in Pure Pd sro
        structure.set_chemical_symbols(['Pd'] * len(structure))
        structure[0].symbol = 'Au'
        counts = observer.get_observable(structure)

        # In total there will be 12 Pd neighboring an Au atom
        expected_Au_Pd_count = 12
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Pd_count, actual_counts)

        # Number of Au-Au neighbors should be
        expected_Au_Au_count = 0
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' not in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Au_count, actual_counts)

        # Number of Pd-Pd neighbors should be
        expected_Au_Au_count = 6 * len(structure) - 12
        actual_counts = 0
        for count in counts.keys():
            if 'Pd' in count and '1' in count and 'Au' not in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Au_count, actual_counts)


class TestClusterCountSpecificOrbitIndices(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterCountSpecificOrbitIndices,
              self).__init__(*args, **kwargs)

        prim = bulk('Au')
        self.structure = prim.repeat(3)
        cutoffs = [6, 5]
        subelements = ['Au', 'Pd']
        self.cs = ClusterSpace(prim, cutoffs, subelements)
        self.interval = 10

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up observer before each test."""
        self.observer_cut = ClusterCountObserver(
            cluster_space=self.cs, structure=self.structure, interval=self.interval,
            orbit_indices=[0, 1, 2])
        self.observer_full = ClusterCountObserver(
            cluster_space=self.cs, structure=self.structure, interval=self.interval)

    def test_failing_initialization(self):
        """
        Tests that intiialization fails when orbit_indices is not a list
        (i.e., the old interface is used)
        """
        with self.assertRaises(ValueError) as cm:
            ClusterCountObserver(self.cs, self.structure, self.interval, 1)
        self.assertIn('should be a list', str(cm.exception))

    def test_get_observable(self):
        """Tests observable is returned accordingly."""
        structure = self.structure.copy()
        structure.set_chemical_symbols(['Au'] * len(structure))
        # 1 Pd in pure Au sro
        structure[0].symbol = 'Pd'

        # Observer with orbit_indices
        counts = self.observer_cut.get_observable(structure)
        self.assertEqual(len(counts), 10)

        # In total there will be 12 Pd neighboring an Au atom
        expected_Au_Pd_count = 12
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Pd_count, actual_counts)

        # Observer without orbit_indices
        counts = self.observer_full.get_observable(structure)
        self.assertEqual(len(counts), 74)

        # In total there will be 12 Pd neighboring an Au atom
        expected_Au_Pd_count = 12
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and count[:2] == '1_' and 'Pd' in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Pd_count, actual_counts)


if __name__ == '__main__':
    unittest.main()
