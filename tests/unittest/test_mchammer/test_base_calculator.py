import unittest
from mchammer.calculators.base_calculator import BaseCalculator


class ConcreteCalculator(BaseCalculator):

    def __init__(self, name='ConcreteCalc'):
        super().__init__(name=name)

    def calculate_total(self):
        super().calculate_total()

    def calculate_change(self):
        super().calculate_change()


class TestBaseCalculator(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestBaseCalculator, self).__init__(*args, **kwargs)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ConcreteCalculator()

    def test_calculate_total(self):
        """Tests calculate total."""
        self.calculator.calculate_total()

    def test_calculate_change(self):
        """Tests calculate local contribution."""
        self.calculator.calculate_change()


if __name__ == '__main__':
    unittest.main()
