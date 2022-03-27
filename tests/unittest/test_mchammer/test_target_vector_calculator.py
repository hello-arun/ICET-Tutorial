import unittest

import numpy as np
from ase.build import bulk
from icet import ClusterSpace
from mchammer.calculators import (TargetVectorCalculator,
                                  compare_cluster_vectors)


class TestTVCalculatorBinary(unittest.TestCase):
    """
    Container for tests of the class functionality.
    """

    def __init__(self, *args, **kwargs):
        super(TestTVCalculatorBinary, self).__init__(*args, **kwargs)

        self.prim = bulk('Al', a=4.0)

        structure = self.prim.repeat((2, 1, 1))
        self.structure = structure.repeat(3)
        self.cutoffs = [5, 5]
        self.elements = ['Al', 'Ge']
        self.cs = ClusterSpace(self.prim, self.cutoffs, self.elements)
        self.target_vector = np.linspace(-1, 1, len(self.cs))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.structure = self.prim.repeat(4)

        self.calculator = TargetVectorCalculator(
            structure=self.structure, cluster_space=self.cs,
            target_vector=self.target_vector,
            name='Tests target vector calc')

    def test_init_with_weights(self):
        """Test init with weights."""
        calculator = TargetVectorCalculator(
            structure=self.structure, cluster_space=self.cs,
            target_vector=self.target_vector,
            weights=np.linspace(3, 1, len(self.cs)),
            optimality_weight=3.0,
            optimality_tol=1e-5,
            name='Tests target vector calc')
        self.assertEqual(type(calculator), TargetVectorCalculator)
        self.assertEqual(calculator.optimality_weight, 3.0)

    def test_init_without_optimality_weight(self):
        """Test init without optimality weight."""
        calculator = TargetVectorCalculator(
            structure=self.structure, cluster_space=self.cs,
            target_vector=self.target_vector,
            optimality_weight=None,
            name='Tests target vector calc')
        self.assertEqual(type(calculator), TargetVectorCalculator)
        self.assertEqual(calculator.optimality_weight, None)

    def test_init_with_erroneous_weights(self):
        """Test init without weights of wrong length."""
        with self.assertRaises(ValueError) as cm:
            TargetVectorCalculator(
                structure=self.structure, cluster_space=self.cs,
                target_vector=self.target_vector,
                weights=[1.0],
                optimality_weight=None,
                name='Tests target vector calc')
        self.assertTrue(
            'Cluster space and weights' in str(cm.exception))

    def test_property_cluster_space(self):
        """Tests the cluster space property."""
        self.assertIsInstance(
            self.calculator.cluster_space, ClusterSpace)

    def test_calculate_total(self):
        """Test calculate_total function."""
        occupations = []
        for i in range(len(self.structure)):
            if i % 2 == 0:
                occupations.append(13)
            else:
                occupations.append(32)
        self.assertAlmostEqual(
            self.calculator.calculate_total(occupations), 7.6363636)


class TestTVCalculatorBinaryHCP(unittest.TestCase):
    """
    Container for tests of the class functionality.
    """

    def __init__(self, *args, **kwargs):
        super(TestTVCalculatorBinaryHCP, self).__init__(*args, **kwargs)

        self.prim = bulk('Al', a=4.0, crystalstructure='hcp')

        structure = self.prim.repeat((2, 1, 1))
        self.structure = structure.repeat(3)
        self.cutoffs = [5, 5]
        self.elements = [['Al', 'Ge'], ['Al', 'Ge']]
        self.cs = ClusterSpace(self.prim, self.cutoffs, self.elements)
        self.target_vector = np.linspace(-1, 1, len(self.cs))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.structure = self.prim.repeat(3)

        self.calculator = TargetVectorCalculator(
            structure=self.structure, cluster_space=self.cs,
            target_vector=self.target_vector,
            optimality_weight=0,
            name='Tests target vector calc')

    def test_calculate_total(self):
        """Test calculate_total function."""
        occupations = []
        for i in range(len(self.structure)):
            if i % 2 == 0:
                occupations.append(32)
            else:
                occupations.append(13)
        self.assertAlmostEqual(
            self.calculator.calculate_total(occupations), 7.0)

    def test_compare_cluster_vectors(self):
        """Test compare_cluster_vector function."""
        orbit_data = self.cs.orbit_data
        cv_1 = np.array(range(0, len(self.cs)))
        cv_2 = cv_1
        optimality_weight = 3.0
        score = compare_cluster_vectors(cv_1, cv_2, orbit_data,
                                        optimality_weight=optimality_weight)
        self.assertAlmostEqual(score, - 2.0 * optimality_weight)

        orbit_data = self.cs.orbit_data
        cv_1 = np.array(range(0, len(self.cs)))
        cv_2 = cv_1
        optimality_weight = None
        score = compare_cluster_vectors(cv_1, cv_2, orbit_data,
                                        weights=np.linspace(3, 5, len(cv_1)),
                                        optimality_weight=optimality_weight)
        self.assertAlmostEqual(score, 0)

        orbit_data = self.cs.orbit_data
        cv_1 = np.array(range(0, len(self.cs)))
        cv_2 = cv_1 + 1.0
        optimality_weight = 2.5
        score = compare_cluster_vectors(cv_1, cv_2, orbit_data,
                                        optimality_weight=optimality_weight)
        self.assertAlmostEqual(score, len(self.cs))


class TestTVCalculatorTernary(unittest.TestCase):
    """
    Container for tests of the class functionality.
    """

    def __init__(self, *args, **kwargs):
        super(TestTVCalculatorTernary, self).__init__(*args, **kwargs)

        self.prim = bulk('Al', a=4.0)

        structure = self.prim.repeat((2, 1, 1))
        self.structure = structure.repeat(3)
        self.cutoffs = [5, 5]
        self.elements = ['Al', 'Ge', 'Ga']
        self.cs = ClusterSpace(self.prim, self.cutoffs, self.elements)
        self.target_vector = np.linspace(-1, 1, len(self.cs))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.structure = self.prim.repeat(4)

        self.calculator = TargetVectorCalculator(
            structure=self.structure, cluster_space=self.cs,
            target_vector=self.target_vector,
            weights=np.linspace(3.1, 1.2, len(self.cs)),
            name='Tests target vector calc')

    def test_property_cluster_space(self):
        """Tests the cluster space property."""
        self.assertIsInstance(
            self.calculator.cluster_space, ClusterSpace)

    def test_calculate_total(self):
        """Test calculate_total function."""
        occupations = []
        for i in range(len(self.structure)):
            if i % 2 == 0:
                occupations.append(13)
            else:
                if i % 3 == 0:
                    occupations.append(32)
                else:
                    occupations.append(31)
        self.assertAlmostEqual(
            self.calculator.calculate_total(occupations), 58.486900169)


if __name__ == '__main__':
    unittest.main()
