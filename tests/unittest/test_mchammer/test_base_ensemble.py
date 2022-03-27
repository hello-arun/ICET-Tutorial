import unittest

import os
import tempfile
import numpy as np
from ase import Atoms
from ase.build import bulk
from pandas.testing import assert_frame_equal

from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators.cluster_expansion_calculator import \
    ClusterExpansionCalculator
from mchammer.ensembles.base_ensemble import BaseEnsemble, dicts_equal
from mchammer.observers.base_observer import BaseObserver
from mchammer.data_containers.base_data_container import BaseDataContainer


class AppleObserver(BaseObserver):
    """Apple says 2.63323e+20."""

    def __init__(self, interval, tag='Apple'):
        super().__init__(interval=interval, return_type=float, tag=tag)

    def get_observable(self, structure):  # noqa
        """Say 2.63323e+20."""
        return 2.63323e+20


class DictObserver(BaseObserver):

    def __init__(self, interval, tag='Ayaymama'):
        super().__init__(interval=interval, return_type=dict, tag=tag)

    def get_observable(self, structure):
        return {'value_1': 1.0, 'value_2': 2.0}

    def get_keys(self):
        return ['value_1', 'value_2']


# Create a concrete child of Ensemble for testing


class ConcreteEnsemble(BaseEnsemble):

    def __init__(self, structure, calculator, temperature=None,
                 user_tag=None, dc_filename=None,
                 data_container_write_period=np.inf, random_seed=None,
                 ensemble_data_write_interval=None,
                 trajectory_write_interval=None):
        self._ensemble_parameters = dict(temperature=temperature)
        super().__init__(
            structure=structure, calculator=calculator, user_tag=user_tag,
            dc_filename=dc_filename,
            data_container_write_period=data_container_write_period,
            random_seed=random_seed,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

    def _do_trial_step(self):
        return 1


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=True)
        self.structure = prim.repeat(3)
        self.cs = ClusterSpace(prim, cutoffs=[1.1], chemical_symbols=['Ag', 'Au'])
        self.ce = ClusterExpansion(self.cs, [0, 0, 2])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.structure, self.ce)
        self.ensemble = ConcreteEnsemble(
            structure=self.structure, calculator=self.calculator, temperature=1000,
            user_tag='test-ensemble', random_seed=42)

        # Create an observer for testing.
        observer = AppleObserver(interval=7)
        self.ensemble.attach_observer(observer)
        observer = AppleObserver(interval=14, tag='Apple2')
        self.ensemble.attach_observer(observer)

    def test_init(self):
        """Tests exceptions are raised in initialization."""

        with self.assertRaises(TypeError) as context:
            ConcreteEnsemble(calculator=self.calculator)
        self.assertIn("required positional argument: 'structure'",
                      str(context.exception))

        with self.assertRaises(TypeError) as context:
            ConcreteEnsemble(structure=self.structure)
        self.assertIn("required positional argument: 'calculator'",
                      str(context.exception))

        # wrong path to data container file
        with self.assertRaises(FileNotFoundError) as context:
            ConcreteEnsemble(structure=self.structure,
                             calculator=self.calculator,
                             dc_filename='path/to/nowhere/mydc')

        self.assertTrue('Path to data container file does not exist:'
                        ' path/to/nowhere' in str(context.exception))

        # wrong occupations on structure
        wrong_structure = self.structure.copy()
        wrong_structure.numbers = [1]*len(self.structure)
        with self.assertRaises(ValueError) as context:
            ConcreteEnsemble(structure=wrong_structure,
                             calculator=self.calculator)

        self.assertTrue('Occupations of structure not compatible with '
                        'the sublattice' in str(context.exception))

    def test_init_fails_for_faulty_chemical_symbols(self):
        """Tests that initialization fails if species exists  on
        multiple sublattices"""
        structure = bulk('Al').repeat(2)
        cutoffs = [4.0]
        elements = [['Al', 'Ga']] * 4 + [['Al', 'Ge']] * 4
        cs = ClusterSpace(structure, cutoffs, elements)
        ce = ClusterExpansion(cs, np.arange(0, len(cs)))
        calc = ClusterExpansionCalculator(structure, ce)
        with self.assertRaises(ValueError) as context:
            ConcreteEnsemble(structure, calc)
        self.assertIn('found on multiple active sublattices', str(context.exception))

    def test_property_user_tag(self):
        """Tests name property."""
        self.assertEqual('test-ensemble', self.ensemble.user_tag)

    def test_property_structure(self):
        """Tests structure property."""
        self.assertEqual(self.structure, self.ensemble.structure)

    def test_property_random_seed(self):
        """Tests random seed property."""
        self.assertEqual(self.ensemble.random_seed, 42)

    def test_property_accepted_trials(self):
        """Tests property accepted trials."""
        self.assertEqual(self.ensemble._accepted_trials, 0)
        self.ensemble._accepted_trials += 1
        self.assertEqual(self.ensemble._accepted_trials, 1)

    def test_property_step(self):
        """Tests property accepted trials."""
        self.assertEqual(self.ensemble._step, 0)
        self.ensemble._step += 1
        self.assertEqual(self.ensemble._step, 1)

    def test_property_calculator(self):
        """Tests the calculator property."""
        pass

    def test_get_next_random_number(self):
        """Tests the get_next_random_number method."""
        self.assertAlmostEqual(
            self.ensemble._next_random_number(), 0.6394267984578837)

    def test_run(self):
        """Tests the run method."""

        n_steps1 = 364
        self.ensemble.run(n_steps1)
        self.assertEqual(self.ensemble._step, n_steps1)

        dc_data = self.ensemble.data_container.get('Apple2')
        number_of_observations = len([x for x in dc_data if x is not None])
        # plus one since we also observe at step 0
        n_target_obs = n_steps1 // self.ensemble.observers['Apple2'].interval + 1
        self.assertEqual(number_of_observations, n_target_obs)

        # run it again to check that step accumulates
        n_steps2 = 68
        self.ensemble.run(n_steps2)
        self.assertEqual(self.ensemble._step, n_steps1+n_steps2)

        dc_data = self.ensemble.data_container.get('Apple2')
        number_of_observations = len([x for x in dc_data if x is not None])
        n_target_obs = (n_steps1 + n_steps2) // self.ensemble.observers['Apple2'].interval + 1
        self.assertEqual(number_of_observations, n_target_obs)

    def test_get_random_sublattice_index(self):
        """Tests the random sublattice index method."""

        with self.assertRaises(ValueError) as context:
            self.ensemble.get_random_sublattice_index([1, 0])
        self.assertIn("probability_distribution should have the same size as sublattices",
                      str(context.exception))

    def test_run_with_dict_observer(self):
        """Tests the run method with a dict observer."""
        observer = DictObserver(interval=28)
        self.ensemble.attach_observer(observer)

        n_iters = 364
        self.ensemble.run(n_iters)
        self.assertEqual(self.ensemble._step, n_iters)
        dc_data = \
            self.ensemble.data_container.get('value_1', 'value_2')

        self.assertEqual(len(dc_data[0]), len(dc_data[1]))

        number_of_observations = len([x for x in dc_data[0] if x is not None])
        # plus one since we also count step 0
        self.assertEqual(
            number_of_observations,
            n_iters // self.ensemble.observers['Ayaymama'].interval + 1)

    def test_backup_file(self):
        """Tests data is being saved and can be read by the ensemble."""
        # set-up ensemble with a non-inf write period
        ensemble = ConcreteEnsemble(structure=self.structure,
                                    calculator=self.calculator,
                                    user_tag='this-ensemble',
                                    dc_filename='my-datacontainer.dc',
                                    data_container_write_period=1e-2,
                                    ensemble_data_write_interval=14,
                                    trajectory_write_interval=56)

        # attach observer
        observer = AppleObserver(interval=14, tag='Apple2')
        ensemble.attach_observer(observer)

        # back-up data while run ensemble and then read the file
        try:
            n_iters = 182
            ensemble.run(n_iters)
            dc_read = BaseDataContainer.read('my-datacontainer.dc')
        finally:
            os.remove('my-datacontainer.dc')

        # check data container
        dc_data = dc_read.get('Apple2')
        self.assertEqual(
            len(dc_data),
            n_iters // observer.interval + 1)

        # write data container to tempfile
        temp_container_file = tempfile.NamedTemporaryFile(delete=False)
        temp_container_file.close()
        dc_read.write(temp_container_file.name)

        # initialise a new ensemble with dc file
        ensemble_reloaded = \
            ConcreteEnsemble(structure=self.structure,
                             calculator=self.calculator,
                             dc_filename=temp_container_file.name,
                             ensemble_data_write_interval=14,
                             trajectory_write_interval=56)

        assert_frame_equal(ensemble.data_container.data,
                           ensemble_reloaded.data_container.data,
                           check_dtype=False,
                           check_like=True)

        # run old and new ensemble and check both data containers are equal
        try:
            n_iters = 50
            ensemble.run(n_iters)
        finally:
            os.remove('my-datacontainer.dc')

        ensemble_reloaded.attach_observer(observer)
        ensemble_reloaded.run(n_iters)

        assert_frame_equal(ensemble.data_container.data,
                           ensemble_reloaded.data_container.data,
                           check_dtype=False,
                           check_like=True)

        self.assertEqual(ensemble_reloaded.data_container._last_state['last_step'], 182 + 50)

    def test_restart_different_parameters(self):
        """Tests that restarting ensemble from data container with different
        ensemble parameters fails."""
        n_iters = 10
        self.ensemble.run(n_iters)
        ensemble_T1000 = tempfile.NamedTemporaryFile(delete=False)
        ensemble_T1000.close()
        self.ensemble.write_data_container(ensemble_T1000.name)

        with self.assertRaises(ValueError) as context:
            ConcreteEnsemble(structure=self.structure,
                             calculator=self.calculator,
                             temperature=3000,
                             dc_filename=ensemble_T1000.name)
        os.remove(ensemble_T1000.name)
        self.assertIn("Ensemble parameters do not match those stored in"
                      " data container file: {('temperature', 1000)}",
                      str(context.exception))

    def test_restart_with_inactive_sites(self):
        """ Tests restart works with inactive sites """

        chemical_symbols = [['C', 'Be'], ['W']]
        prim = bulk('W', 'bcc', cubic=True, )
        cs = ClusterSpace(structure=prim, chemical_symbols=chemical_symbols, cutoffs=[5])
        parameters = [1] * len(cs)
        ce = ClusterExpansion(cs, parameters)

        size = 4
        structure = ce._cluster_space.primitive_structure.repeat(size)
        calculator = ClusterExpansionCalculator(structure, ce)

        # Carry out Monte Carlo simulations
        dc_file = tempfile.NamedTemporaryFile(delete=False)
        dc_file.close()
        mc = ConcreteEnsemble(structure=structure, calculator=calculator)
        mc.write_data_container(dc_file.name)
        mc.run(10)

        # and now restart
        mc = ConcreteEnsemble(structure=structure, calculator=calculator,
                              dc_filename=dc_file.name)
        mc.run(10)
        os.remove(dc_file.name)

    def test_internal_run(self):
        """Tests the _run method."""
        pass

    def test_attach_observer(self):
        """Tests the attach method."""
        self.assertEqual(len(self.ensemble.observers), 2)

        self.ensemble.attach_observer(
            AppleObserver(interval=10, tag='test_Apple'))
        self.assertEqual(self.ensemble.observers['test_Apple'].interval, 10)
        self.assertEqual(
            self.ensemble.observers['test_Apple'].tag, 'test_Apple')
        self.assertEqual(len(self.ensemble.observers), 3)

        # test no duplicates, this should overwrite the last Apple
        self.ensemble.attach_observer(
            AppleObserver(interval=15), tag='test_Apple')
        self.assertEqual(len(self.ensemble.observers), 3)
        self.assertEqual(self.ensemble.observers['test_Apple'].interval, 15)
        self.assertEqual(
            self.ensemble.observers['test_Apple'].tag, 'test_Apple')

        # check that correct exceptions are raised
        with self.assertRaises(TypeError) as context:
            self.ensemble.attach_observer('xyz')
        self.assertTrue('observer has the wrong type'
                        in str(context.exception))

    def test_property_data_container(self):
        """Tests the data container property."""
        self.assertIsInstance(self.ensemble.data_container, BaseDataContainer)

    def test_find_minimum_observation_interval(self):
        """Tests the method to find the minimum observation interval."""
        pass

    def test_property_minimum_observation_interval(self):
        """Tests property minimum observation interval."""
        pass

    def test_get_gcd(self):
        """Tests the get gcd method."""
        input = [2, 4, 6, 8]
        target = 2
        self.assertEqual(self.ensemble._get_gcd(input), target)

        input = [20, 15, 10]
        target = 5
        self.assertEqual(self.ensemble._get_gcd(input), target)

        input = [1]
        target = 1
        self.assertEqual(self.ensemble._get_gcd(input), target)

    def test_get_property_change(self):
        """Tests the get property change method."""

        initial_occupations = self.ensemble.configuration.occupations

        indices = [0, 1, 2, 3, 4]
        elements = [79, 79, 47, 79, 79]

        prop_diff = self.ensemble._get_property_change(indices, elements)
        self.assertAlmostEqual(prop_diff, -8)

        # Tests that the method doesn't change the occupation.
        self.assertListEqual(list(initial_occupations),
                             list(self.ensemble.configuration.occupations))

        with self.assertRaises(ValueError) as context:
            self.ensemble.update_occupations(indices, elements + [79])

        self.assertTrue('sites and species must have the same length.'
                        in str(context.exception))

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()
        self.assertIn('potential', data.keys())

    def test_non_default_metadata(self):
        """Tests metadata generated by ensemble."""
        self.assertEqual('test-ensemble',
                         self.ensemble.data_container.metadata['user_tag'])
        self.assertEqual(
            'ConcreteEnsemble',
            self.ensemble.data_container.metadata['ensemble_name'])
        self.assertEqual(
            'ConcreteEnsemble',
            self.ensemble.data_container.metadata['ensemble_name'])

    def test_dicts_equal(self):
        """Tests dicts_equal function."""
        d1 = dict(T=300.25, phi=-0.1, kappa=200)
        d2 = {k: v for k, v in d1.items()}

        # check dicts are equal
        self.assertTrue(dicts_equal(d1, d2))

        # check dicts are equal even with small difference
        d2['T'] += 1e-16
        self.assertTrue(dicts_equal(d1, d2))

        # check dicts differ when a larger difference is introduced
        d2['T'] += 1e-10
        self.assertFalse(dicts_equal(d1, d2))

    def test_str(self):
        """Tests __str__ method."""
        self.ensemble.run(10)
        ret = str(self.ensemble)
        self.assertIsInstance(ret, str)


if __name__ == '__main__':
    unittest.main()
