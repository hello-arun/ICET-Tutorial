import unittest

import numpy as np
from ase import Atom
from ase.build import bulk

from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator

from mchammer.ensembles import VCSGCEnsemble
from mchammer.ensembles.vcsgc_ensemble import get_phis


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
        self.phis = {'Al': -1.3}
        self.kappa = 10.0
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

        self.ensemble = VCSGCEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            temperature=self.temperature,
            phis=self.phis,
            kappa=self.kappa,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(TypeError) as context:
            VCSGCEnsemble(structure=self.structure, calculator=self.calculator)
        self.assertIn("required positional arguments: 'temperature'", str(context.exception))

        with self.assertRaises(TypeError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature)
        self.assertIn("required positional arguments: 'phis'", str(context.exception))

        with self.assertRaises(TypeError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis=self.phis)
        self.assertIn("required positional argument: 'kappa'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={'Al': -2.0, 'Ga': 0.0},
                          kappa=self.kappa)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

    def test_property_phis(self):
        """Tests phis property."""
        retval = self.ensemble.phis
        target = {13: -1.3}
        self.assertEqual(retval, target)

    def test_get_phis(self):
        """Tests get_phis function"""
        retval = get_phis({'Al': -1.2})
        target = {13: -1.2}
        self.assertEqual(retval, target)

        retval = get_phis({'Ga': -1.2})
        target = {31: -1.2}
        self.assertEqual(retval, target)

        with self.assertRaises(TypeError) as context:
            get_phis('xyz')
        self.assertIn('phis has the wrong type', str(context.exception))

    def test_run(self):
        """Test that run function runs. """
        n = 50
        self.ensemble.run(n)
        self.assertEqual(self.ensemble.step, n)

    def test_property_boltzmann(self):
        """Tests explicit Boltzmann constant."""
        self.assertAlmostEqual(1e-5, self.ensemble.boltzmann_constant)

    def test_property_temperature(self):
        """Tests temperature property."""
        self.assertEqual(self.ensemble.temperature, self.temperature)

    def test_property_kappa(self):
        """Tests kappa property."""
        self.assertEqual(self.ensemble.kappa, self.kappa)

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

    def test_init_with_integer_phis(self):
        """Tests init with integer chemical potentials."""

        phis = {13: -1}
        ensemble = VCSGCEnsemble(
            structure=self.structure, calculator=self.calculator,
            user_tag='test-ensemble', random_seed=42,
            temperature=self.temperature,
            phis=phis,
            kappa=self.kappa)
        ensemble._do_trial_step()

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Al_count', data.keys())
        self.assertIn('Ga_count', data.keys())

        self.assertEqual(data['Al_count'], 13)
        self.assertEqual(data['Ga_count'], 14)

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'], len(self.structure))
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'], self.temperature)
        self.assertEqual(self.ensemble.ensemble_parameters['phi_Al'], -1.3)
        self.assertEqual(self.ensemble.ensemble_parameters['kappa'], 10)

        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['temperature'],
                         self.temperature)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['phi_Al'], -1.3)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['kappa'], 10)

    def test_write_interval_and_period(self):
        """
        Tests interval and period for writing data from ensemble.
        """
        self.assertEqual(self.ensemble._data_container_write_period, 499.0)
        self.assertEqual(self.ensemble._ensemble_data_write_interval, 25)
        self.assertEqual(self.ensemble._trajectory_write_interval, 40)


class TestEnsembleTernaryFCC(unittest.TestCase):
    """Container for tests of the class functionality for ternary system."""

    def __init__(self, *args, **kwargs):
        super(TestEnsembleTernaryFCC, self).__init__(*args, **kwargs)

        self.structure = bulk('Al').repeat(3)
        for i, atom in enumerate(self.structure):
            if i % 2 == 0:
                atom.symbol = 'Ga'
            elif i % 3 == 0:
                atom.symbol = 'In'
        cutoffs = [5, 5, 4]
        elements = ['Al', 'Ga', 'In']
        self.phis = {'Al': -1.3, 'Ga': -0.4}
        self.kappa = 10.0
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

        self.ensemble = VCSGCEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            temperature=self.temperature,
            phis=self.phis,
            kappa=self.kappa,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={'Al': -0.5},
                          kappa=self.kappa)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={'Al': -0.5, 'Ga': -0.3, 'In': -0.7},
                          kappa=self.kappa)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

    def test_property_phis(self):
        """Tests phis property."""
        retval = self.ensemble.phis
        target = {13: -1.3, 31: -0.4}
        self.assertEqual(retval, target)

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

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Al_count', data.keys())
        self.assertIn('Ga_count', data.keys())
        self.assertIn('In_count', data.keys())

        self.assertEqual(data['Al_count'], 9)
        self.assertEqual(data['Ga_count'], 14)
        self.assertEqual(data['In_count'], 4)

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'], len(self.structure))
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'], self.temperature)
        self.assertEqual(self.ensemble.ensemble_parameters['phi_Al'], -1.3)
        self.assertEqual(self.ensemble.ensemble_parameters['phi_Ga'], -0.4)
        self.assertEqual(self.ensemble.ensemble_parameters['kappa'], 10)

        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['temperature'],
                         self.temperature)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['phi_Al'], -1.3)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['phi_Ga'], -0.4)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['kappa'], 10)


class TestEnsembleSublattices(unittest.TestCase):
    """
    Container for tests of the class functionality for a system
    with two sublattices.
    """

    def __init__(self, *args, **kwargs):
        super(TestEnsembleSublattices, self).__init__(*args, **kwargs)

        lattice_parameter = 4.0
        prim = bulk('Pd', a=lattice_parameter, crystalstructure='fcc')
        prim.append(Atom('H', position=(lattice_parameter / 2,)*3))
        self.structure = prim.repeat(3)
        for i, atom in enumerate(self.structure):
            if i % 3 == 0:
                if atom.symbol == 'Pd':
                    atom.symbol = 'Au'
                else:
                    atom.symbol = 'V'
        cutoffs = [5, 5, 4]
        elements = [['Pd', 'Au'], ['H', 'V']]
        self.phis = {'Au': -1.3, 'H': -1.0}
        self.kappa = 200.0
        self.cs = ClusterSpace(prim, cutoffs, elements)
        parameters = parameters = np.array([1.2] * len(self.cs))
        self.ce = ClusterExpansion(self.cs, parameters)
        self.temperature = 100.0

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.structure, self.ce)

        self.ensemble = VCSGCEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            temperature=self.temperature,
            phis=self.phis,
            kappa=self.kappa,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={'Pd': -2.0, 'Au': 0.0},
                          kappa=self.kappa)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={'Au': -2.0},
                          kappa=self.kappa)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={'Au': -2.0, 'H': -1.2, 'V': -1.0},
                          kappa=self.kappa)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={},
                          kappa=self.kappa)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

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

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Au_count', data.keys())
        self.assertIn('Pd_count', data.keys())
        self.assertIn('H_count', data.keys())
        self.assertIn('V_count', data.keys())
        self.assertIn('free_energy_derivative_Au', data.keys())
        self.assertIn('free_energy_derivative_H', data.keys())
        self.assertNotIn('free_energy_derivative_Pd', data.keys())
        self.assertNotIn('free_energy_derivative_V', data.keys())

        self.assertEqual(data['H_count'], 18)
        self.assertEqual(data['V_count'], 9)

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'], len(self.structure))
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'], self.temperature)
        self.assertAlmostEqual(self.ensemble.ensemble_parameters['phi_Au'], -1.3)
        self.assertAlmostEqual(self.ensemble.ensemble_parameters['phi_H'], -1.0)
        self.assertAlmostEqual(self.ensemble.ensemble_parameters['kappa'], 200.0)

        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['temperature'],
                         self.temperature)
        self.assertAlmostEqual(self.ensemble.data_container.ensemble_parameters['phi_Au'], -1.3)
        self.assertAlmostEqual(self.ensemble.data_container.ensemble_parameters['phi_H'], -1.0)
        self.assertAlmostEqual(self.ensemble.data_container.ensemble_parameters['kappa'], 200.0)

    def test_write_interval_and_period(self):
        """
        Tests interval and period for writing data from ensemble.
        """
        self.assertEqual(self.ensemble._data_container_write_period, 499)
        self.assertEqual(self.ensemble._ensemble_data_write_interval, 25)
        self.assertEqual(self.ensemble._trajectory_write_interval, 40)


class TestEnsembleSpectatorSublattice(unittest.TestCase):
    """
    Container for tests of the class functionality for a system
    with two sublattices.
    """

    def __init__(self, *args, **kwargs):
        super(TestEnsembleSpectatorSublattice, self).__init__(*args, **kwargs)

        lattice_parameter = 4.0
        prim = bulk('Pd', a=lattice_parameter, crystalstructure='fcc')
        prim.append(Atom('H', position=(lattice_parameter / 2,)*3))
        self.structure = prim.repeat(3)
        for i, atom in enumerate(self.structure):
            if i % 3 == 0:
                if atom.symbol == 'Pd':
                    continue
                else:
                    atom.symbol = 'V'
        cutoffs = [5, 5, 4]
        elements = [['Pd'], ['H', 'V']]
        self.phis = {'H': -1.0}
        self.kappa = 200.0
        self.cs = ClusterSpace(prim, cutoffs, elements)
        parameters = parameters = np.array([1.2] * len(self.cs))
        self.ce = ClusterExpansion(self.cs, parameters)
        self.temperature = 100.0

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.structure, self.ce)

        self.ensemble = VCSGCEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            temperature=self.temperature,
            phis=self.phis,
            kappa=self.kappa,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(structure=self.structure,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={'Pd': -2.0},
                          kappa=self.kappa)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

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

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Pd_count', data.keys())
        self.assertIn('H_count', data.keys())
        self.assertIn('V_count', data.keys())
        self.assertIn('free_energy_derivative_H', data.keys())
        self.assertNotIn('free_energy_derivative_Pd', data.keys())
        self.assertNotIn('free_energy_derivative_V', data.keys())

        self.assertEqual(data['H_count'], 18)
        self.assertEqual(data['V_count'], 9)

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'], len(self.structure))
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'], self.temperature)
        self.assertAlmostEqual(self.ensemble.ensemble_parameters['phi_H'], -1.0)
        self.assertAlmostEqual(self.ensemble.ensemble_parameters['kappa'], 200.0)

        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['temperature'],
                         self.temperature)
        self.assertAlmostEqual(self.ensemble.data_container.ensemble_parameters['phi_H'], -1.0)
        self.assertAlmostEqual(self.ensemble.data_container.ensemble_parameters['kappa'], 200.0)


if __name__ == '__main__':
    unittest.main()
