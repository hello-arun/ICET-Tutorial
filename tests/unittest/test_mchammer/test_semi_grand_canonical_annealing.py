import unittest

import numpy as np
from ase.build import bulk

from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator

from mchammer.ensembles.sgc_annealing import SGCAnnealing, available_cooling_functions


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        # setup supercell
        self.structure = bulk('Al').repeat(3)
        for i, atom in enumerate(self.structure):
            if i % 2 == 0:
                atom.symbol = 'Ga'

        # setup cluster expansion
        cutoffs = [5, 5, 4]
        elements = ['Al', 'Ga']
        cs = ClusterSpace(self.structure, cutoffs, elements)
        parameters = parameters = np.array([1.2] * len(cs))
        self.ce = ClusterExpansion(cs, parameters)
        self.chemical_potentials = {'Al': 1, 'Ga': 2}

        self.T_start = 1000.0
        self.T_stop = 0.0
        self.n_steps = 500

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.structure, self.ce)
        self.ensemble = SGCAnnealing(
            structure=self.structure,
            calculator=self.calculator,
            chemical_potentials=self.chemical_potentials,
            T_start=self.T_start,
            T_stop=self.T_stop,
            n_steps=self.n_steps,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_different_cooling_functions(self):
        """ Tests that different cooling functions works """

        # make sure init works with available cooling functions
        for f_name in available_cooling_functions.keys():
            mc = SGCAnnealing(
                chemical_potentials=self.chemical_potentials,
                structure=self.structure, calculator=self.calculator, T_start=self.T_start,
                T_stop=self.T_stop, n_steps=self.n_steps, cooling_function=f_name)
            mc.run()

        # make sure init works with user-defined cooling function
        def best_annealing_function(step, T_start, T_stop, n_steps):
            return np.random.uniform(T_stop, T_start)
        mc = SGCAnnealing(
            chemical_potentials=self.chemical_potentials,
            structure=self.structure, calculator=self.calculator, T_start=self.T_start,
            T_stop=self.T_stop, n_steps=self.n_steps, cooling_function=best_annealing_function)
        mc.run()

    def test_run(self):
        """ Tests that run works and raises if annealing is finished """
        self.ensemble.run()

        with self.assertRaises(Exception) as context:
            self.ensemble.run()
        self.assertIn('Annealing has already finished', str(context.exception))

    def test_do_trial_step(self):
        """Tests the do trial step."""

        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""

        # always accept negative delta potential
        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()
        self.assertIn('temperature', data.keys())
        self.assertIn('potential', data.keys())

        self.assertIn('Al_count', data.keys())
        self.assertIn('Ga_count', data.keys())

        self.assertEqual(data['Al_count'], 13)
        self.assertEqual(data['Ga_count'], 14)

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_steps'], self.n_steps)

    def test_estimated_ground_state_properties(self):
        """ Tests the estimated ground state properites."""
        self.ensemble.run()
        traj, potential = self.ensemble.data_container.get('trajectory', 'potential')
        min_ind = potential.argmin()

        self.assertEqual(traj[min_ind], self.ensemble.estimated_ground_state)
        self.assertAlmostEqual(potential[min_ind], self.ensemble.estimated_ground_state_potential,
                               places=12)


if __name__ == '__main__':
    unittest.main()
