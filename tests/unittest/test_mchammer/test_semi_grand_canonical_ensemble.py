import unittest
import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
from mchammer.ensembles.semi_grand_canonical_ensemble import get_chemical_potentials


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        self.structure = bulk('Al').repeat(3)
        for i, atom in enumerate(self.structure):
            if i % 2 == 0:
                atom.symbol = 'Ga'
        cutoffs = [5, 5, 4]
        elements = ['Al', 'Ga']
        self.chemical_potentials = {'Al': 5, 'Ga': 0}
        self.cs = ClusterSpace(self.structure, cutoffs, elements)
        parameters = parameters = np.array([1.2] * len(self.cs))
        self.ce = ClusterExpansion(self.cs, parameters)
        self.temperature = 100.0

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.structure, self.ce)

        self.ensemble = SemiGrandCanonicalEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40,
            temperature=self.temperature,
            chemical_potentials=self.chemical_potentials,
            boltzmann_constant=1e-5)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(TypeError) as context:
            SemiGrandCanonicalEnsemble(structure=self.structure,
                                       calculator=self.calculator)
        self.assertTrue("required positional arguments: 'temperature'" in
                        str(context.exception))

        with self.assertRaises(TypeError) as context:
            SemiGrandCanonicalEnsemble(structure=self.structure,
                                       calculator=self.calculator,
                                       temperature=self.temperature)
        self.assertTrue("required positional argument:"
                        " 'chemical_potentials'" in str(context.exception))

    def test_property_boltzmann(self):
        """Tests explicit Boltzmann constant."""
        self.assertAlmostEqual(1e-5, self.ensemble.boltzmann_constant)

    def test_property_temperature(self):
        """Tests property temperature."""
        self.assertEqual(self.ensemble.temperature, self.temperature)

    def test_property_chemical_potentials(self):
        """Tests property chemical_potentials."""
        retval = self.ensemble.chemical_potentials
        target = {13: 5, 31: 0}
        self.assertEqual(retval, target)

        # test exceptions
        with self.assertRaises(TypeError) as context:
            get_chemical_potentials('xyz')
        self.assertTrue('chemical_potentials has the wrong type'
                        in str(context.exception))

    def test_run(self):
        """Test that run function runs. """
        n = 50
        self.ensemble.run(n)
        self.assertEqual(self.ensemble.step, n)

    def test_do_trial_step(self):
        """Tests the do trial step."""
        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""
        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_init_with_integer_chemical_potentials(self):
        """Tests init with integer chemical potentials."""

        chemical_potentials = {13: 5, 31: 0}
        ensemble = SemiGrandCanonicalEnsemble(
            structure=self.structure, calculator=self.calculator,
            user_tag='test-ensemble',
            random_seed=42, temperature=self.temperature,
            chemical_potentials=chemical_potentials)
        ensemble._do_trial_step()

        # Test both int and str
        chemical_potentials = {'Al': 5, 31: 0}
        ensemble = SemiGrandCanonicalEnsemble(
            structure=self.structure, calculator=self.calculator,
            user_tag='test-ensemble',
            random_seed=42, temperature=self.temperature,
            chemical_potentials=chemical_potentials)
        ensemble._do_trial_step()

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Al_count', data.keys())
        self.assertIn('Ga_count', data.keys())

        self.assertEqual(data['Al_count'], 13)
        self.assertEqual(data['Ga_count'], 14)

    def test_get_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'],
                         self.temperature)
        self.assertEqual(self.ensemble.ensemble_parameters['mu_Al'], 5)
        self.assertEqual(self.ensemble.ensemble_parameters['mu_Ga'], 0)

        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['n_atoms'],
            len(self.structure))
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['temperature'],
            self.temperature)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['mu_Al'], 5)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['mu_Ga'], 0)

    def test_write_interval_and_period(self):
        """Tests interval and period for writing data from ensemble."""
        self.assertEqual(self.ensemble._data_container_write_period, 499.0)
        self.assertEqual(self.ensemble._ensemble_data_write_interval, 25)
        self.assertEqual(self.ensemble._trajectory_write_interval, 40)


if __name__ == '__main__':
    unittest.main()
