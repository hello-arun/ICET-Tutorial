import unittest

import numpy as np
from ase import Atoms
from ase.build import bulk

from icet import ClusterSpace
from mchammer.calculators import TargetVectorCalculator
from mchammer.ensembles import TargetClusterVectorAnnealing


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        # setup supercells
        self.prim = bulk('Al')
        self.structure = []
        structure = self.prim.repeat(4)
        for i, atom in enumerate(structure):
            if i % 2 == 0:
                atom.symbol = 'Ga'
        self.structure.append(structure)
        structure = self.prim.repeat(2)
        for i, atom in enumerate(structure):
            if i % 2 == 0:
                atom.symbol = 'Ga'
        self.structure.append(structure)

        # setup cluster expansion
        cutoffs = [5, 5, 4]
        elements = ['Al', 'Ga']
        cs = ClusterSpace(self.prim, cutoffs, elements)

        target_vector = np.linspace(-1, 1, len(cs))

        self.calculators = []
        for structure in self.structure:
            self.calculators.append(TargetVectorCalculator(structure,
                                                           cs,
                                                           target_vector))

        self.T_start = 3.0
        self.T_stop = 0.01

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.ensemble = TargetClusterVectorAnnealing(
            structure=self.structure,
            calculators=self.calculators,
            T_start=self.T_start,
            T_stop=self.T_stop,
            random_seed=42)

    def test_init_raises_if_wrong_type(self):
        """Test that ensemble cannot be initialized with ASE Atoms."""
        with self.assertRaises(ValueError) as cm:
            TargetClusterVectorAnnealing(
                structure=bulk('Al'),
                calculators=self.calculators,
                T_start=self.T_start,
                T_stop=self.T_stop)
        self.assertTrue(
            'A list of ASE Atoms (supercells)' in str(cm.exception))

    def test_init_raises_if_not_equal_list_lengths(self):
        """Test that ensemble cannot be init with unequal length lists."""
        with self.assertRaises(ValueError) as cm:
            TargetClusterVectorAnnealing(
                self.structure,
                calculators=self.calculators[:-1],
                T_start=self.T_start,
                T_stop=self.T_stop)
        self.assertTrue(
            'There must be as many supercells' in str(cm.exception))

    def test_init_without_random_seed(self):
        """Test that init without random seed specification works."""
        ensemble = TargetClusterVectorAnnealing(
            structure=self.structure,
            calculators=self.calculators,
            T_start=self.T_start,
            T_stop=self.T_stop)
        self.assertEqual(type(ensemble._random_seed), int)

    def test_generate_structure(self):
        """ Tests that run works and raises if annealing is finished """
        structure = self.ensemble.generate_structure(number_of_trial_steps=13)
        self.assertEqual(self.ensemble.total_trials, 13)
        self.assertIsInstance(structure, Atoms)

    def test_do_trial_step(self):
        """Tests the do trial step."""

        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()
        self.assertEqual(self.ensemble.total_trials, 10)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""

        # always accept negative delta potential
        self.assertTrue(self.ensemble._acceptance_condition(-1.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(1.0)


if __name__ == '__main__':
    unittest.main()
