import unittest

from ase.build import bulk
from icet import ClusterSpace, ClusterExpansion
from mchammer.observers.cluster_expansion_observer import \
    ClusterExpansionObserver


class TestCEObserver(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestCEObserver, self).__init__(*args, **kwargs)

        self.structure = bulk('Al').repeat(3)

        cutoffs = [6, 6, 5]
        subelements = ['Al', 'Ge']
        cs = ClusterSpace(self.structure, cutoffs, subelements)
        params_len = len(cs)
        params = list(range(params_len))

        self.ce = ClusterExpansion(cs, params)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up observer before each test."""
        self.observer = ClusterExpansionObserver(
            self.ce, tag='ce_band_gap', interval=10)

    def test_property_tag(self):
        """Tests property tag."""
        self.assertEqual(self.observer.tag, 'ce_band_gap')

    def test_property_interval(self):
        """Tests property interval."""
        self.assertEqual(self.observer.interval, 10)

    def test_get_observable(self):
        """Tests observable is returned accordingly."""
        self.assertEqual(self.observer.get_observable(structure=self.structure), 283.0)

        # updated occupations and check again
        structure = self.structure.copy()
        indices = [2, 4, 10]
        for ind in indices:
            structure[ind].symbol = 'Ge'
        self.assertAlmostEqual(self.observer.get_observable(structure=structure), 1808.0 / 27)


if __name__ == '__main__':
    unittest.main()
