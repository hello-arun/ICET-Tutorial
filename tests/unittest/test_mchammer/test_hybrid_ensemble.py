import unittest

import numpy as np
from ase import Atom
from ase.build import bulk

from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator

from mchammer.ensembles import HybridEnsemble
from mchammer.ensembles.vcsgc_ensemble import get_phis
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
        self.phis = {'Al': -1.3}
        self.kappa = 10.0
        self.ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0},
                               {'ensemble': 'semi-grand', 'sublattice_index': 0,
                               'chemical_potentials': self.chemical_potentials},
                               {'ensemble': 'vcsgc', 'sublattice_index': 0,
                               'phis': self.phis, 'kappa': self.kappa}]
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

        self.ensemble = HybridEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            ensemble_specs=self.ensemble_specs,
            temperature=self.temperature,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(TypeError) as context:
            HybridEnsemble(structure=self.structure, calculator=self.calculator)
        self.assertIn("required positional arguments: 'temperature'", str(context.exception))

        with self.assertRaises(TypeError) as context:
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature)
        self.assertIn("required positional argument: 'ensemble_specs'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The dictionary {} lacks the required key"
                      " 'ensemble'".format(ensemble_specs[0]), str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical'}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The dictionary {} lacks the key 'sublattice_index', which is required for"
                      " {} ensembles".format(ensemble_specs[0], ensemble_specs[0]['ensemble']),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0, 'temperature': 100}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("Unknown key 'temperature', for a {} ensemble, in the dictionary"
                      " {}".format(ensemble_specs[0]['ensemble'], ensemble_specs[0]),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'grand', 'sublattice_index': 0}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('Ensemble not available\nPlease choose one of the following:\n * canonical'
                      '\n * semi-grand\n * vcsgc', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'semi-grand', 'sublattice_index': 0}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The dictionary {} lacks the key 'chemical_potentials', which is required for"
                      " {} ensembles".format(ensemble_specs[0], ensemble_specs[0]['ensemble']),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The dictionary {} lacks the key 'phis', which is required for {}"
                      " ensembles".format(ensemble_specs[0], ensemble_specs[0]['ensemble']),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0, 'phis': self.phis}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The dictionary {} lacks the key 'kappa', which is required for {}"
                      " ensembles".format(ensemble_specs[0], ensemble_specs[0]['ensemble']),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0, 'phis': self.phis}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("Unknown key 'phis', for a {} ensemble, in the dictionary"
                      " {}".format(ensemble_specs[0]['ensemble'], ensemble_specs[0]),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("Unknown key 'kappa', for a {} ensemble, in the dictionary"
                      " {}".format(ensemble_specs[0]['ensemble'], ensemble_specs[0]),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0,
                               'chemical_potentials': self.chemical_potentials}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("Unknown key 'chemical_potentials', for a {} ensemble, in the dictionary"
                      " {}".format(ensemble_specs[0]['ensemble'], ensemble_specs[0]),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'semi-grand', 'sublattice_index': 0,
                               'chemical_potentials': self.chemical_potentials, 'phis': self.phis}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("Unknown key 'phis', for a {} ensemble, in the dictionary"
                      " {}".format(ensemble_specs[0]['ensemble'], ensemble_specs[0]),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'semi-grand', 'sublattice_index': 0,
                               'chemical_potentials': self.chemical_potentials,
                               'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("Unknown key 'kappa', for a {} ensemble, in the dictionary"
                      " {}".format(ensemble_specs[0]['ensemble'], ensemble_specs[0]),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0,
                               'phis': self.phis, 'kappa': self.kappa,
                               'chemical_potentials': self.chemical_potentials}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("Unknown key 'chemical_potentials', for a {} ensemble, in the dictionary "
                      "{}".format(ensemble_specs[0]['ensemble'], ensemble_specs[0]),
                      str(context.exception))

        with self.assertRaises(TypeError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 'A'}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("'sublattice_index' must be an integer, not {}".format(type('A')),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 1}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("There is no sublattice with index 1", str(context.exception))

        with self.assertRaises(TypeError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0,
                               'allowed_symbols': 'Al'}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("'allowed_symbols' must be a List[str], not {}".format(type('Al')),
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0,
                               'allowed_symbols': ['Ge']}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The species Ge is not allowed on sublattice 0",
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0,
                               'phis': {'Al': -2.0, 'Ga': 0.0}, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs,
                           probabilities=[])
        self.assertIn('The number of probabilities must be match the number of ensembles',
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs,
                           probabilities=[1.1])
        self.assertIn('The sum of all probabilities must be equal to 1', str(context.exception))

    def test_ensemble_args(self):
        """Tests the ensemble args."""
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            self.assertEqual(self.ensemble._ensemble_args[i]['ensemble'], ensemble_spec['ensemble'])
            self.assertEqual(self.ensemble._ensemble_args[i]['sublattice_index'], 0)
            self.assertIsNone(self.ensemble._ensemble_args[i]['allowed_species'])
        self.assertDictEqual(self.ensemble._ensemble_args[1]['chemical_potentials'],
                             get_chemical_potentials(self.chemical_potentials))
        self.assertDictEqual(self.ensemble._ensemble_args[2]['phis'], get_phis(self.phis))
        self.assertEqual(self.ensemble._ensemble_args[2]['kappa'], self.kappa)

    def test_trial_steps_per_ensemble(self):
        """Tests trial steps per ensemble property."""
        retval = self.ensemble.trial_steps_per_ensemble
        target = {'ensemble_{}'.format(i): 0 for i in range(len(self.ensemble_specs))}
        self.assertDictEqual(retval, target)

    def test_property_boltzmann(self):
        """Tests explicit Boltzmann constant."""
        self.assertAlmostEqual(1e-5, self.ensemble.boltzmann_constant)

    def test_property_temperature(self):
        """Tests temperature property."""
        self.assertEqual(self.ensemble.temperature, self.temperature)

    def test_property_probabilities(self):
        """Tests probabilities property."""
        self.assertEqual(self.ensemble.probabilities,
                         [1/len(self.ensemble_specs)]*len(self.ensemble_specs))

    def test_do_trial_step(self):
        """Tests the do trial step."""
        # Do it many times and hopefully get both a reject and an accept
        steps = 100
        for _ in range(steps):
            self.ensemble._do_trial_step()

        # Check that all steps were performed with all ensembles
        for retval in self.ensemble.trial_steps_per_ensemble.values():
            self.assertGreater(retval, 0)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""

        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_init_with_integer_chemical_potentials(self):
        """Tests init with integer chemical potentials."""

        ensemble_specs = [{'ensemble': 'semi-grand', 'sublattice_index': 0,
                           'chemical_potentials': {13: 5, 31: 0}}]
        ensemble = HybridEnsemble(
                       structure=self.structure,
                       calculator=self.calculator,
                       temperature=self.temperature,
                       ensemble_specs=ensemble_specs,
                       user_tag='test-ensemble',
                       random_seed=42,)
        ensemble._do_trial_step()

        # Test both int and str
        ensemble_specs = [{'ensemble': 'semi-grand', 'sublattice_index': 0,
                           'chemical_potentials': {'Al': 5, 31: 0}}]
        ensemble = HybridEnsemble(
                       structure=self.structure,
                       calculator=self.calculator,
                       temperature=self.temperature,
                       ensemble_specs=ensemble_specs,
                       user_tag='test-ensemble',
                       random_seed=42,)
        ensemble._do_trial_step()

    def test_init_with_integer_phis(self):
        """Tests init with integer phis."""
        ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0,
                           'phis': {13: -1}, 'kappa': self.kappa}]
        ensemble = HybridEnsemble(
                       structure=self.structure,
                       calculator=self.calculator,
                       temperature=self.temperature,
                       ensemble_specs=ensemble_specs,
                       user_tag='test-ensemble',
                       random_seed=42,)
        ensemble._do_trial_step()

    def test_init_with_probabilities(self):
        """Tests init with probabilities."""
        probabilities = [1.0/3.0, 0.5, 1/6]
        ensemble = HybridEnsemble(
                       structure=self.structure,
                       calculator=self.calculator,
                       temperature=self.temperature,
                       ensemble_specs=self.ensemble_specs,
                       probabilities=probabilities,
                       user_tag='test-ensemble',
                       random_seed=42,)

        # Test the probabilities property
        self.assertEqual(ensemble.probabilities, probabilities)

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
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            tag = 'ensemble_{}'.format(i)
            self.assertEqual(self.ensemble.ensemble_parameters[tag], ensemble_spec['ensemble'])
            if ensemble_spec['ensemble'] == 'semi-grand':
                for sym, mu in self.chemical_potentials.items():
                    self.assertEqual(
                        self.ensemble.ensemble_parameters['{}_mu_{}'.format(tag, sym)], mu)
            if ensemble_spec['ensemble'] == 'vcsgc':
                for sym, phi in self.phis.items():
                    self.assertEqual(
                        self.ensemble.ensemble_parameters['{}_phi_{}'.format(tag, sym)], phi)
                self.assertEqual(self.ensemble.ensemble_parameters['{}_kappa'.format(tag)],
                                 self.kappa)

        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['temperature'],
                         self.temperature)
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            tag = 'ensemble_{}'.format(i)
            self.assertEqual(self.ensemble.data_container.ensemble_parameters[tag],
                             ensemble_spec['ensemble'])
            if ensemble_spec['ensemble'] == 'semi-grand':
                for sym, mu in self.chemical_potentials.items():
                    self.assertEqual(
                        self.ensemble.data_container.ensemble_parameters[
                            '{}_mu_{}'.format(tag, sym)], mu)
            if ensemble_spec['ensemble'] == 'vcsgc':
                for sym, phi in self.phis.items():
                    self.assertEqual(
                        self.ensemble.data_container.ensemble_parameters[
                            '{}_phi_{}'.format(tag, sym)], phi)
                self.assertEqual(self.ensemble.data_container.ensemble_parameters[
                                    '{}_kappa'.format(tag)], self.kappa)

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
        self.chemical_potentials = {'Al': 5, 'Ga': 0, 'In': 1}
        self.phis = {'Al': -1.3, 'Ga': -0.4}
        self.kappa = 10.0
        self.ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0,
                                'allowed_symbols': elements[:-1]},
                               {'ensemble': 'semi-grand', 'sublattice_index': 0,
                                'chemical_potentials': self.chemical_potentials},
                               {'ensemble': 'vcsgc', 'sublattice_index': 0,
                                'phis': self.phis, 'kappa': self.kappa}]
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

        self.ensemble = HybridEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            ensemble_specs=self.ensemble_specs,
            temperature=self.temperature,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0,
                               'phis': {'Al': -0.5}, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0,
                               'phis': {'Al': -0.5, 'Ga': -0.3, 'In': -0.7}, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

    def test_ensemble_args(self):
        """Tests the ensemble args."""
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            self.assertEqual(self.ensemble._ensemble_args[i]['ensemble'], ensemble_spec['ensemble'])
            self.assertEqual(self.ensemble._ensemble_args[i]['sublattice_index'], 0)
            if i == 0:
                self.assertListEqual(self.ensemble._ensemble_args[i]['allowed_species'],
                                     [13, 31])
            else:
                self.assertIsNone(self.ensemble._ensemble_args[i]['allowed_species'])
        self.assertDictEqual(self.ensemble._ensemble_args[1]['chemical_potentials'],
                             get_chemical_potentials(self.chemical_potentials))
        self.assertDictEqual(self.ensemble._ensemble_args[2]['phis'], get_phis(self.phis))
        self.assertEqual(self.ensemble._ensemble_args[2]['kappa'], self.kappa)

    def test_do_trial_step(self):
        """Tests the do trial step."""
        # Do it many times and hopefully get both a reject and an accept
        steps = 100
        for _ in range(steps):
            self.ensemble._do_trial_step()

        # Check that all steps were performed with all ensembles
        for retval in self.ensemble.trial_steps_per_ensemble.values():
            self.assertGreater(retval, 0)

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
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            tag = 'ensemble_{}'.format(i)
            self.assertEqual(self.ensemble.ensemble_parameters[tag], ensemble_spec['ensemble'])
            if ensemble_spec['ensemble'] == 'semi-grand':
                for sym, mu in self.chemical_potentials.items():
                    self.assertEqual(
                        self.ensemble.ensemble_parameters['{}_mu_{}'.format(tag, sym)], mu)
            if ensemble_spec['ensemble'] == 'vcsgc':
                for sym, phi in self.phis.items():
                    self.assertEqual(
                        self.ensemble.ensemble_parameters['{}_phi_{}'.format(tag, sym)], phi)
                self.assertEqual(self.ensemble.ensemble_parameters['{}_kappa'.format(tag)],
                                 self.kappa)

        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['temperature'],
                         self.temperature)
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            tag = 'ensemble_{}'.format(i)
            self.assertEqual(self.ensemble.data_container.ensemble_parameters[tag],
                             ensemble_spec['ensemble'])
            if ensemble_spec['ensemble'] == 'semi-grand':
                for sym, mu in self.chemical_potentials.items():
                    self.assertEqual(
                        self.ensemble.data_container.ensemble_parameters[
                            '{}_mu_{}'.format(tag, sym)], mu)
            if ensemble_spec['ensemble'] == 'vcsgc':
                for sym, phi in self.phis.items():
                    self.assertEqual(
                        self.ensemble.data_container.ensemble_parameters[
                            '{}_phi_{}'.format(tag, sym)], phi)
                self.assertEqual(self.ensemble.data_container.ensemble_parameters[
                                    '{}_kappa'.format(tag)], self.kappa)


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
        self.chemical_potentials = {'H': 5, 'V': 0}
        self.phis = {'Au': -1.3}
        self.kappa = 200.0
        self.ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0},
                               {'ensemble': 'semi-grand', 'sublattice_index': 1,
                                'chemical_potentials': self.chemical_potentials},
                               {'ensemble': 'vcsgc', 'sublattice_index': 0, 'phis': self.phis,
                                'kappa': self.kappa}]
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

        self.ensemble = HybridEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            ensemble_specs=self.ensemble_specs,
            temperature=self.temperature,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0,
                               'phis': {'Pd': -2.0, 'Au': 0.0}, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0,
                               'phis': {}, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 1,
                               'phis': {'H': -2.0, 'V': 0.0}, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 1,
                               'phis': {}, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 2}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("There is no sublattice with index 2", str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0,
                              'allowed_symbols': ['H']}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The species H is not allowed on sublattice 0",
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 1,
                              'allowed_symbols': ['Au']}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The species Au is not allowed on sublattice 1",
                      str(context.exception))

    def test_property_probabilities(self):
        """Tests probabilities property."""
        self.assertEqual(self.ensemble.probabilities,
                         [1/len(self.ensemble_specs)]*len(self.ensemble_specs))

    def test_do_trial_step(self):
        """Tests the do trial step."""
        # Do it many times and hopefully get both a reject and an accept
        steps = 100
        for _ in range(steps):
            self.ensemble._do_trial_step()

        # Check that all steps were performed with all ensembles
        for retval in self.ensemble.trial_steps_per_ensemble.values():
            self.assertGreater(retval, 0)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""

        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_init_with_probabilities(self):
        """Tests init with probabilities."""
        probabilities = [1.0/3.0, 0.5, 1/6]
        ensemble = HybridEnsemble(
                       structure=self.structure,
                       calculator=self.calculator,
                       temperature=self.temperature,
                       ensemble_specs=self.ensemble_specs,
                       probabilities=probabilities,
                       user_tag='test-ensemble',
                       random_seed=42,)

        # Test the probabilities property
        self.assertEqual(ensemble.probabilities, probabilities)

        ensemble._do_trial_step()

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Au_count', data.keys())
        self.assertIn('Pd_count', data.keys())
        self.assertIn('H_count', data.keys())
        self.assertIn('V_count', data.keys())
        self.assertIn('free_energy_derivative_Au', data.keys())
        self.assertNotIn('free_energy_derivative_H', data.keys())
        self.assertNotIn('free_energy_derivative_Pd', data.keys())
        self.assertNotIn('free_energy_derivative_V', data.keys())

        self.assertEqual(data['H_count'], 18)
        self.assertEqual(data['V_count'], 9)

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'], len(self.structure))
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'], self.temperature)
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            tag = 'ensemble_{}'.format(i)
            self.assertEqual(self.ensemble.ensemble_parameters[tag], ensemble_spec['ensemble'])
            if ensemble_spec['ensemble'] == 'semi-grand':
                for sym, mu in self.chemical_potentials.items():
                    self.assertEqual(
                        self.ensemble.ensemble_parameters['{}_mu_{}'.format(tag, sym)], mu)
            if ensemble_spec['ensemble'] == 'vcsgc':
                for sym, phi in self.phis.items():
                    self.assertEqual(
                        self.ensemble.ensemble_parameters['{}_phi_{}'.format(tag, sym)], phi)
                self.assertEqual(self.ensemble.ensemble_parameters['{}_kappa'.format(tag)],
                                 self.kappa)

        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['temperature'],
                         self.temperature)
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            tag = 'ensemble_{}'.format(i)
            self.assertEqual(self.ensemble.data_container.ensemble_parameters[tag],
                             ensemble_spec['ensemble'])
            if ensemble_spec['ensemble'] == 'semi-grand':
                for sym, mu in self.chemical_potentials.items():
                    self.assertEqual(
                        self.ensemble.data_container.ensemble_parameters[
                            '{}_mu_{}'.format(tag, sym)], mu)
            if ensemble_spec['ensemble'] == 'vcsgc':
                for sym, phi in self.phis.items():
                    self.assertEqual(
                        self.ensemble.data_container.ensemble_parameters[
                            '{}_phi_{}'.format(tag, sym)], phi)
                self.assertEqual(self.ensemble.data_container.ensemble_parameters[
                                    '{}_kappa'.format(tag)], self.kappa)

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
        self.chemical_potentials = {'H': 5, 'V': 0}
        self.phis = {'H': -1.0}
        self.kappa = 200.0
        self.ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0},
                               {'ensemble': 'semi-grand', 'sublattice_index': 0,
                                'chemical_potentials': self.chemical_potentials},
                               {'ensemble': 'vcsgc', 'sublattice_index': 0,
                                'phis': self.phis, 'kappa': self.kappa}]
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

        self.ensemble = HybridEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            ensemble_specs=self.ensemble_specs,
            temperature=self.temperature,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'vcsgc', 'sublattice_index': 0,
                               'phis': {'Pd': -2.0}, 'kappa': self.kappa}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn('phis must be set for N - 1 elements', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 1}]
            HybridEnsemble(structure=self.structure,
                           calculator=self.calculator,
                           temperature=self.temperature,
                           ensemble_specs=ensemble_specs)
        self.assertIn("The sublattice 1 is inactive", str(context.exception))

    def test_property_probabilities(self):
        """Tests probabilities property."""
        self.assertEqual(self.ensemble.probabilities,
                         [1/len(self.ensemble_specs)]*len(self.ensemble_specs))

    def test_do_trial_step(self):
        """Tests the do trial step."""
        # Do it many times and hopefully get both a reject and an accept
        steps = 100
        for _ in range(steps):
            self.ensemble._do_trial_step()

        # Check that all steps were performed with all ensembles
        for retval in self.ensemble.trial_steps_per_ensemble.values():
            self.assertGreater(retval, 0)

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
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            tag = 'ensemble_{}'.format(i)
            self.assertEqual(self.ensemble.ensemble_parameters[tag], ensemble_spec['ensemble'])
            if ensemble_spec['ensemble'] == 'semi-grand':
                for sym, mu in self.chemical_potentials.items():
                    self.assertEqual(
                        self.ensemble.ensemble_parameters['{}_mu_{}'.format(tag, sym)], mu)
            if ensemble_spec['ensemble'] == 'vcsgc':
                for sym, phi in self.phis.items():
                    self.assertEqual(
                        self.ensemble.ensemble_parameters['{}_phi_{}'.format(tag, sym)], phi)
                self.assertEqual(self.ensemble.ensemble_parameters['{}_kappa'.format(tag)],
                                 self.kappa)

        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'],
                         len(self.structure))
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['temperature'],
                         self.temperature)
        for i, ensemble_spec in enumerate(self.ensemble_specs):
            tag = 'ensemble_{}'.format(i)
            self.assertEqual(self.ensemble.data_container.ensemble_parameters[tag],
                             ensemble_spec['ensemble'])
            if ensemble_spec['ensemble'] == 'semi-grand':
                for sym, mu in self.chemical_potentials.items():
                    self.assertEqual(
                        self.ensemble.data_container.ensemble_parameters[
                            '{}_mu_{}'.format(tag, sym)], mu)
            if ensemble_spec['ensemble'] == 'vcsgc':
                for sym, phi in self.phis.items():
                    self.assertEqual(
                        self.ensemble.data_container.ensemble_parameters[
                            '{}_phi_{}'.format(tag, sym)], phi)
                self.assertEqual(self.ensemble.data_container.ensemble_parameters[
                                    '{}_kappa'.format(tag)], self.kappa)


if __name__ == '__main__':
    unittest.main()
