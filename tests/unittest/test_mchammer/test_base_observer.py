import unittest

from mchammer.observers.base_observer import BaseObserver
from ase.build import bulk


class TestBaseObserver(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestBaseObserver, self).__init__(*args, **kwargs)
        self.structure = bulk('Al').repeat(3)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        # Create a concrete child of BaseObserver for testing
        class ConcreteObserver(BaseObserver):

            def __init__(self, interval, tag='ConcreteObserver'):
                super().__init__(interval=interval, return_type=int, tag=tag)

            def get_observable(self, structure):
                """
                Return number of Al structure.
                """
                return structure.get_chemical_symbols().count('Al')

        self.observer = ConcreteObserver(interval=10, tag='test_observer')

    def test_return_type(self):
        """Tests property return type."""
        self.assertEqual(self.observer.return_type, int)

    def test_get_observable(self):
        """Tests base observer by calling concrete observers get_observable."""
        self.assertEqual(self.observer.get_observable(self.structure), 27)

    def test_interval_attribute(self):
        """Tests interval attribute."""
        self.assertEqual(self.observer.interval, 10)

    def test_attribute_tag(self):
        """Tests the tag attribute."""
        self.assertEqual(self.observer.tag, 'test_observer')


if __name__ == '__main__':
    unittest.main()
