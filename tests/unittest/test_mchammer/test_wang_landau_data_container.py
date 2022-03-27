import os
import random
import unittest
from collections import OrderedDict
import warnings

import numpy as np
from ase import Atoms
from pandas import DataFrame

from icet import ClusterSpace
from mchammer.data_containers.wang_landau_data_container import (WangLandauDataContainer,
                                                                 get_density_of_states_wl,
                                                                 get_average_observables_wl,
                                                                 get_average_cluster_vectors_wl)
from mchammer.observers.base_observer import BaseObserver


def _assertAlmostEqualList(self, retval, target, places=6):
    """
    Helper function that conducts an element-wise comparison of two lists.
    """
    self.assertIsInstance(retval, type(target))
    self.assertEqual(len(retval), len(target))
    for k, (r, t) in enumerate(zip(retval, target)):
        s = ['element: {}'.format(k)]
        s += ['retval: {} ({})'.format(r, type(r))]
        s += ['target: {} ({})'.format(t, type(t))]
        info = '   '.join(s)
        self.assertAlmostEqual(r, t, places=places, msg=info)


unittest.TestCase.assertAlmostEqualList = _assertAlmostEqualList


class ConcreteObserver(BaseObserver):
    """Child class of BaseObserver created for testing."""
    def __init__(self, interval, tag='ConcreteObserver'):
        super().__init__(interval=interval, return_type=int, tag=tag)

    def get_observable(self, structure):
        """Returns number of Au atoms."""
        return structure.get_chemical_symbols().count('Au')


class TestDataContainer(unittest.TestCase):
    """Container for the tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestDataContainer, self).__init__(*args, **kwargs)
        # further settings
        self.prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
        self.fill_factor_history = OrderedDict((10 * k, 1 / 2 ** k) for k in range(7))
        self.entropy_history = {10 * j: OrderedDict((k, 1 / (1 + k**2 * (j + 1))) for k in
                                                    range(-20, 21)) for j in range(7)}
        self.entropy = OrderedDict((k, 1 / (1 + k**2)) for k in range(-20, 21))
        self.histogram = OrderedDict((k, int(k / 2 + 3)) for k in range(-20, 21))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def prepareDataContainer(self,
                             energy_limit_left=None, energy_limit_right=None,
                             energy_spacing=1, n_rep=3, trial_move='swap',
                             fill_factor=1e-6):
        """This method prepares a single data container for testing.
        """
        def split_dict(data, left_limit, right_limit):
            return OrderedDict((k, v) for k, v in data.items()
                               if (left_limit is None or k > left_limit) and
                               (right_limit is None or k < right_limit))

        # prepare initial configuration
        structure = self.prim.repeat(n_rep)
        structure[0].symbol = 'Ag'
        structure[1].symbol = 'Ag'
        structure = structure.repeat(n_rep)
        n_atoms = len(structure)

        # compiling data container
        dc = WangLandauDataContainer(
            structure=structure,
            ensemble_parameters={'n_atoms': len(structure),
                                 'energy_spacing': energy_spacing,
                                 'energy_limit_left': energy_limit_left,
                                 'energy_limit_right': energy_limit_right,
                                 'trial_move': trial_move})
        dc._update_last_state(
            last_step=206,
            occupations=[9] * len(structure),
            accepted_trials=10,
            random_state=random.getstate(),
            fill_factor=fill_factor,
            fill_factor_history=self.fill_factor_history,
            entropy_history={mctrial: split_dict(entropy, energy_limit_left, energy_limit_right)
                             for mctrial, entropy in self.entropy_history.items()},
            histogram=split_dict(self.histogram, energy_limit_left, energy_limit_right),
            entropy=split_dict(self.entropy, energy_limit_left, energy_limit_right))
        data_rows = OrderedDict([
            (0, {'potential': -1.32,
                 'obs': 1,
                 'occupations': [79 if k < 0.8*n_atoms else 47 for k in range(n_atoms)]}),
            (10, {'potential': -1.35}),
            (20, {'potential': -1.33,
                  'obs': 2,
                  'occupations': [79 if k < 0.1*n_atoms else 47 for k in range(n_atoms)]}),
            (30, {'potential': -1.07}),
            (40, {'potential': -1.02,
                  'obs': 3,
                  'occupations': [79 if k < 0.6*n_atoms else 47 for k in range(n_atoms)]}),
            (50, {'potential': -1.4}),
            (60, {'potential': -1.3,
                  'obs': 4,
                  'occupations': [79 if k < 0.5*n_atoms else 47 for k in range(n_atoms)]})])

        for mctrial in data_rows:
            dc.append(mctrial, data_rows[mctrial])

        return dc

    def prepare_dc_fill_factor_limit_ref(self, energy_limit_left=None, energy_limit_right=None,
                                         energy_spacing=1, n_rep=3, trial_move='swap',
                                         fill_factor=1e-6):
        """This method prepares a single data container that can be used for
        checking the entropy history."""
        def split_dict(data, left_limit, right_limit):
            return OrderedDict((k, v) for k, v in data.items()
                               if (left_limit is None or k > left_limit) and
                               (right_limit is None or k < right_limit))

        # prepare initial configuration
        structure = self.prim.repeat(n_rep)
        structure[0].symbol = 'Ag'
        structure[1].symbol = 'Ag'
        structure = structure.repeat(n_rep)
        n_atoms = len(structure)

        # compiling data container
        dc = WangLandauDataContainer(
            structure=structure,
            ensemble_parameters={'n_atoms': len(structure),
                                 'energy_spacing': energy_spacing,
                                 'energy_limit_left': energy_limit_left,
                                 'energy_limit_right': energy_limit_right,
                                 'trial_move': trial_move})
        dc._update_last_state(
            last_step=206,
            occupations=[9] * len(structure),
            accepted_trials=10,
            random_state=random.getstate(),
            fill_factor=fill_factor,
            fill_factor_history=self.fill_factor_history,
            entropy_history={mctrial: split_dict(entropy, energy_limit_left, energy_limit_right)
                             for mctrial, entropy in self.entropy_history.items()},
            histogram=split_dict(self.histogram, energy_limit_left, energy_limit_right),
            entropy=split_dict(self.entropy_history[30], energy_limit_left, energy_limit_right))
        data_rows = OrderedDict([
            (0, {'potential': -1.32,
                 'obs': 1,
                 'occupations': [79 if k < 0.8*n_atoms else 47 for k in range(n_atoms)]}),
            (10, {'potential': -1.35}),
            (20, {'potential': -1.33,
                  'obs': 2,
                  'occupations': [79 if k < 0.1*n_atoms else 47 for k in range(n_atoms)]}),
            (30, {'potential': -1.07})])
        for mctrial in data_rows:
            dc.append(mctrial, data_rows[mctrial])

        return dc

    def test_update_last_state(self):
        """Tests update_last_state functionality."""
        fill_factor = 0.78
        dc = self.prepareDataContainer(fill_factor=fill_factor)
        for key, value in dc._last_state.items():
            if key == 'fill_factor':
                self.assertIsInstance(value, float)
                self.assertAlmostEqual(value, fill_factor)
            if key == 'fill_factor_history':
                self.assertIsInstance(value, dict)
                self.assertDictEqual(value, self.fill_factor_history)
            if key == 'histogram':
                self.assertIsInstance(value, dict)
                self.assertDictEqual(value, self.histogram)
            if key == 'entropy':
                self.assertIsInstance(value, dict)
                self.assertDictEqual(value, self.entropy)

    def test_property_fill_factor(self):
        """Tests fill_factor property."""
        fill_factor = 1e-6
        dc = self.prepareDataContainer(fill_factor=fill_factor)
        self.assertEqual(dc.fill_factor, fill_factor)
        with self.assertRaises(AttributeError) as context:
            dc.fill_factor = 1000
        self.assertTrue("can't set attribute" in str(context.exception))

    def test_property_fill_factor_history(self):
        """Tests fill_factor_history property."""
        dc = self.prepareDataContainer()
        self.assertIsInstance(dc.fill_factor_history, DataFrame)
        self.assertEqual(list(dc.fill_factor_history.fill_factor),
                         list(self.fill_factor_history.values()))
        self.assertEqual(list(dc.fill_factor_history.mctrial),
                         list(self.fill_factor_history.keys()))
        with self.assertRaises(AttributeError) as context:
            dc.fill_factor_history = 1000
        self.assertTrue("can't set attribute" in str(context.exception))

    def test_get(self):
        """ Tests the get function. """

        dc = self.prepareDataContainer()

        # dummy trajectory
        traj = []
        for occupations in DataFrame.from_records(
                dc._data_list, index='mctrial').occupations.dropna():
            atoms = dc.structure.copy()
            atoms.numbers = occupations
            traj.append(atoms)

        # assert numpy array is returned
        mctrial, potential = dc.get('mctrial', 'potential')
        self.assertIsInstance(mctrial, np.ndarray)
        self.assertIsInstance(potential, np.ndarray)

        # with a given fill factor limit
        fill_factor_limit = (self.fill_factor_history[20] + self.fill_factor_history[30]) / 2
        mctrial, obs = dc.get('mctrial', 'obs', fill_factor_limit=fill_factor_limit)
        self.assertEqual(mctrial.tolist(), [0, 20])
        self.assertEqual(obs.tolist(), [1, 2])

        # get trajectory
        traj_ret = dc.get('trajectory')
        self.assertEqual(traj, traj_ret)

        # get trajectory with a given fill factor limit
        fill_factor_limit = (self.fill_factor_history[20] + self.fill_factor_history[30]) / 2
        traj_ret = dc.get('trajectory', fill_factor_limit=fill_factor_limit)
        for i in range(2):
            self.assertEqual(traj[i], traj_ret[i])

        # get trajectory with a given fill factor limit and property
        fill_factor_limit = (self.fill_factor_history[40] + self.fill_factor_history[50]) / 2
        traj_ret, obs1 = dc.get('trajectory', 'obs', fill_factor_limit=fill_factor_limit)
        for i in range(3):
            self.assertEqual(traj[i], traj_ret[i])
        self.assertEqual(obs1.tolist(), [1, 2, 3])

    def test_get_entropy(self):
        """Tests get_entropy method."""
        # test data container without entropy
        dc = self.prepareDataContainer()
        del dc._last_state['entropy']
        with self.assertWarns(Warning) as context:
            entropy = dc.get_entropy()
        self.assertIn('There is no entropy information in the data container',
                      str(context.warning))
        self.assertIsNone(entropy)

        # test without fill factor limit
        dc = self.prepareDataContainer()
        ret = dc.get_entropy()
        self.assertIsInstance(ret, DataFrame)
        ret = ret.sort_index()

        ret_energy = ret.energy.tolist()
        target_energy = list(self.entropy.keys())
        self.assertEqual(ret_energy, target_energy)

        ret_entropy = ret.entropy.tolist()
        target_entropy = np.array(list(self.entropy.values()))
        target_entropy -= np.min(target_entropy)
        target_entropy = target_entropy.tolist()
        self.assertAlmostEqualList(ret_entropy, target_entropy)

        # test with fill factor limit
        fill_factor_limit = (self.fill_factor_history[20] + self.fill_factor_history[30]) / 2
        dc = self.prepareDataContainer()
        ret = dc.get_entropy(fill_factor_limit)
        self.assertIsInstance(ret, DataFrame)
        ret = ret.sort_index()

        ret_energy = ret.energy.tolist()
        target_energy = list(self.entropy_history[30].keys())
        self.assertEqual(ret_energy, target_energy)

        ret_entropy = ret.entropy.tolist()
        target_entropy = np.array(list(self.entropy_history[30].values()))
        target_entropy -= np.min(target_entropy)
        target_entropy = target_entropy.tolist()
        self.assertAlmostEqualList(ret_entropy, target_entropy)

        # test data container with fill factor limit but empty entropy history
        dc = self.prepareDataContainer()
        dc._last_state['entropy_history'] = {}
        with self.assertWarns(Warning) as context:
            ret = dc.get_entropy(fill_factor_limit)
        self.assertIn('The entropy history is empty', str(context.warning))
        self.assertIsNone(ret)

        # test data container with fill factor limit but no entropy history
        dc = self.prepareDataContainer()
        del dc._last_state['entropy_history']
        with self.assertWarns(Warning) as context:
            ret = dc.get_entropy(fill_factor_limit)
        self.assertIn('The entropy history is empty', str(context.warning))
        self.assertIsNone(ret)

        # test data container with fill factor above limit
        dc = self.prepareDataContainer()
        dc._last_state['fill_factor'] = fill_factor_limit * 2
        with self.assertWarns(Warning) as context:
            ret = dc.get_entropy(fill_factor_limit)
        self.assertIn('is higher than the limit', str(context.warning))
        self.assertIsNone(ret)

    def test_get_histogram(self):
        """Tests get_histogram method."""
        dc = self.prepareDataContainer()
        del dc._last_state['histogram']
        self.assertIsNone(dc.get_histogram())

        dc = self.prepareDataContainer()
        ret = dc.get_histogram()
        self.assertIsInstance(ret, DataFrame)
        ret = ret.sort_index()

        ret_energy = ret.energy.tolist()
        target_energy = list(self.histogram.keys())
        self.assertEqual(ret_energy, target_energy)

        ret_histogram = ret.histogram.tolist()
        target_histogram = list(self.histogram.values())
        self.assertEqual(ret_histogram, target_histogram)

    def test_read_write(self):
        """ Tests read/write functionality. """
        file_name = 'my-test.dc'
        dc1 = self.prepareDataContainer()
        dc1.write(file_name)
        dc2 = WangLandauDataContainer.read(file_name)
        self.assertIsInstance(dc2, WangLandauDataContainer)
        for key, val in dc1._last_state.items():
            self.assertIn(key, dc2._last_state)
            if isinstance(val, str) or isinstance(val, int) or \
                    (isinstance(val, list) and isinstance(val[0], int)):
                self.assertEqual(val, dc2._last_state[key])
            elif isinstance(val, float) or \
                    (isinstance(val, list) and isinstance(val[0], float)):
                self.assertAlmostEqual(val, dc2._last_state[key])
            elif isinstance(val, dict) and isinstance(list(val.values())[0], float):
                self.assertAlmostEqualList(list(val.values()), list(dc2._last_state[key].values()))
        self.assertEqual(len(dc1.data), len(dc2.data))
        for col in dc1.data.columns:
            self.assertIn(col, dc2.data.columns)
            data1 = dc1.data[col].dropna().tolist()
            data2 = dc2.data[col].dropna().tolist()
            self.assertAlmostEqualList(data1, data2)
        os.remove(file_name)

    def test_get_density_of_states_wl(self):
        """Tests get_density_of_states function."""

        # test type warning
        with self.assertRaises(TypeError) as context:
            ret = get_density_of_states_wl('abc')
        self.assertTrue('must be a data container with entropy data' in str(context.exception))

        # test warning for data container(s) without entropy
        dc = self.prepareDataContainer()
        del dc._last_state['entropy']
        with self.assertRaises(ValueError) as context:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ret = get_density_of_states_wl(dc)
        self.assertTrue('Entropy information could not be retrieved' in str(context.exception))
        dcs = {'a': self.prepareDataContainer(), 'b': dc}
        with self.assertRaises(ValueError) as context:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ret = get_density_of_states_wl(dcs)
        self.assertTrue('Entropy information could not be retrieved' in str(context.exception))

        # test warning concerning underconverged data with single container
        dc = self.prepareDataContainer(fill_factor=0.1)
        with self.assertWarns(Warning) as context:
            get_density_of_states_wl(dc)
        self.assertIn('underconverged Wang-Landau', str(context.warning))

        # test single container without fill factor limit
        dc = self.prepareDataContainer()
        ret = get_density_of_states_wl(dc)
        self.assertTrue(len(ret), 2)
        self.assertIsInstance(ret[0], DataFrame)
        self.assertIsNone(ret[1])
        ret_single = ret[0]
        for key in ['energy', 'entropy', 'density']:
            self.assertIn(key, ret_single.columns)
        ret_entropy = ret_single.entropy.tolist()
        target_entropy = np.array(list(self.entropy.values()))
        target_entropy -= np.min(target_entropy)
        target_entropy = target_entropy.tolist()
        self.assertAlmostEqualList(ret_entropy, target_entropy)

        # test warning concerning underconverged data with multiple containers
        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=4),
               2: self.prepareDataContainer(energy_limit_left=-4, energy_limit_right=None,
                                            fill_factor=0.1)}
        with self.assertWarns(Warning) as context:
            get_density_of_states_wl(dcs)
        self.assertIn('underconverged Wang-Landau', str(context.warning))

        # test warnings in case of inconsistent containers
        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=4),
               2: self.prepareDataContainer(energy_limit_left=-4, energy_limit_right=None,
                                            n_rep=4)}
        with self.assertRaises(ValueError) as context:
            ret = get_density_of_states_wl(dcs)
        self.assertTrue('Number of atoms differs between data containers' in str(context.exception))

        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=4),
               2: self.prepareDataContainer(energy_limit_left=-4, energy_limit_right=None,
                                            energy_spacing=2)}
        with self.assertRaises(ValueError) as context:
            ret = get_density_of_states_wl(dcs)
        self.assertTrue('energy_spacing differs between data containers' in str(context.exception))

        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=4),
               2: self.prepareDataContainer(energy_limit_left=-4, energy_limit_right=None,
                                            trial_move='flip')}
        with self.assertRaises(ValueError) as context:
            ret = get_density_of_states_wl(dcs)
        self.assertTrue('trial_move differs between data containers' in str(context.exception))

        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=-1),
               2: self.prepareDataContainer(energy_limit_left=1, energy_limit_right=None)}
        with self.assertRaises(ValueError) as context:
            ret = get_density_of_states_wl(dcs)
        self.assertTrue('No overlap in the energy range' in str(context.exception))

        # test overlapping data containers without fill factor limit
        dcs = {}
        for k, (lim1, lim2) in enumerate(zip([None, -12, -2, 8], [-8, 2, 12, None])):
            dcs[k] = self.prepareDataContainer(energy_limit_left=lim1, energy_limit_right=lim2)
        ret = get_density_of_states_wl(dcs)
        self.assertTrue(len(ret), 2)
        self.assertIsInstance(ret[0], DataFrame)
        self.assertIsInstance(ret[1], dict)
        ret_binned = ret[0]
        for key in ['energy', 'entropy', 'density']:
            self.assertIn(key, ret_single.columns)

        ret_entropy = ret_single.entropy.tolist()
        target_entropy = np.array(list(self.entropy.values()))
        target_entropy -= np.min(target_entropy)
        target_entropy = target_entropy.tolist()
        self.assertAlmostEqualList(ret_entropy, target_entropy)

        self.assertEqual(len(ret_binned), len(ret_single))

        # test single container with fill factor limit
        dc = self.prepareDataContainer()
        fill_factor_limit = (self.fill_factor_history[20] + self.fill_factor_history[30]) / 2
        ret = get_density_of_states_wl(dc, fill_factor_limit)
        self.assertTrue(len(ret), 2)
        self.assertIsInstance(ret[0], DataFrame)
        self.assertIsNone(ret[1])
        ret_single = ret[0]
        for key in ['energy', 'entropy', 'density']:
            self.assertIn(key, ret_single.columns)

        ret_entropy = ret_single.entropy.tolist()
        target_entropy = np.array(list(self.entropy_history[30].values()))
        target_entropy -= np.min(target_entropy)
        target_entropy = target_entropy.tolist()
        self.assertAlmostEqualList(ret_entropy, target_entropy)

        # test overlapping data containers with fill factor limit
        fill_factor_limit = (self.fill_factor_history[20] + self.fill_factor_history[30]) / 2
        dcs = {}
        for k, (lim1, lim2) in enumerate(zip([None, -12, -2, 8], [-8, 2, 12, None])):
            dcs[k] = self.prepareDataContainer(energy_limit_left=lim1, energy_limit_right=lim2)
        ret = get_density_of_states_wl(dcs, fill_factor_limit)
        self.assertTrue(len(ret), 2)
        self.assertIsInstance(ret[0], DataFrame)
        self.assertIsInstance(ret[1], dict)
        ret_binned = ret[0]
        for key in ['energy', 'entropy', 'density']:
            self.assertIn(key, ret_binned.columns)

        ret_entropy = ret_binned.entropy.tolist()
        target_entropy = np.array(list(self.entropy_history[30].values()))
        target_entropy -= np.min(target_entropy)
        target_entropy = target_entropy.tolist()
        self.assertAlmostEqualList(ret_entropy, target_entropy)

        self.assertEqual(len(ret_binned), len(ret_single))

        # test data containers with uneven overlap, 1 subset of 2
        dcs = {1: self.prepareDataContainer(energy_limit_left=-12, energy_limit_right=-5),
               2: self.prepareDataContainer(energy_limit_left=-15, energy_limit_right=0)}
        with self.assertWarns(Warning) as context:
            ret = get_density_of_states_wl(dcs, fill_factor_limit)
        self.assertIn('Window 1 is a subset of 2',
                      str(context.warning))
        self.assertTrue(len(ret), 2)
        self.assertIsInstance(ret[0], DataFrame)
        self.assertIsInstance(ret[1], dict)
        ret_binned = ret[0]
        self.assertFalse(all(ret_binned.entropy.isna()))
        for key in ['energy', 'entropy', 'density']:
            self.assertIn(key, ret_binned.columns)

        ret_entropy = ret_binned.entropy.tolist()
        target_entropy = np.array([e for k, e in self.entropy_history[30].items()
                                   if k > -15 and k < 0])
        target_entropy -= np.min(target_entropy)
        target_entropy = target_entropy.tolist()
        self.assertAlmostEqualList(ret_entropy, target_entropy)

    def test_get_average_observables_wl(self):
        """Tests get_average_observables_wl function."""

        temperatures = [500, 1000, 1500, 3000, 5000]
        target = {'temperature': list(temperatures),
                  'potential_mean': [-19.999999999917, -19.999990872733, -19.999563028446,
                                     -19.978651625821, -19.891091736169],
                  'potential_std': [9.128661021746e-06, 3.021150487412e-03, 2.090843168801e-02,
                                    1.476623564612e-01, 3.475208248792e-01],
                  'obs_mean': [0.534371029993, 0.936885807201, 1.092738394744,
                               1.256638406965, 1.324497567828],
                  'obs_std': [1.067669317756, 1.333510197404, 1.402693434773,
                              1.459270899411, 1.477504027232]}

        # test type warning
        with self.assertRaises(TypeError) as context:
            ret = get_average_observables_wl('abc', temperatures)
        self.assertTrue('must be a data container with entropy data' in str(context.exception))

        # test single container without observables
        dc = self.prepareDataContainer()
        ret = get_average_observables_wl(dc, temperatures)
        np.set_printoptions(precision=12)
        self.assertIsInstance(ret, DataFrame)
        for key in ['temperature', 'potential_mean', 'potential_std']:
            self.assertIn(key, ret.columns)
        for col in ret.columns:
            diff = (np.array(ret[col].tolist()) - target[col]) / target[col]
            self.assertTrue(np.all(np.abs(diff) < 1e-6))

        # test single container with observable
        dc = self.prepareDataContainer()
        ret = get_average_observables_wl(dc, temperatures, observables=['obs'])
        np.set_printoptions(precision=12)
        self.assertIsInstance(ret, DataFrame)
        for key in ['temperature', 'potential_mean', 'potential_std', 'obs_mean', 'obs_std']:
            self.assertIn(key, ret.columns)
        for col in ret.columns:
            diff = (np.array(ret[col].tolist()) - target[col]) / target[col]
            self.assertTrue(np.all(np.abs(diff) < 1e-6))

        # test warning in case of unknown observable
        obs_name = 'Swoop'
        with self.assertRaises(ValueError) as context:
            ret = get_average_observables_wl(dc, temperatures, observables=[obs_name])
        self.assertTrue('Observable ({}) not in data'.format(obs_name) in str(context.exception))

        # run with multiple containers
        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=4),
               2: self.prepareDataContainer(energy_limit_left=-4, energy_limit_right=None)}
        ret = get_average_observables_wl(dcs, temperatures, observables=['obs'])
        self.assertIsInstance(ret, DataFrame)
        for key in ['temperature', 'potential_mean', 'potential_std', 'obs_mean', 'obs_std']:
            self.assertIn(key, ret.columns)
        for col in ret.columns:
            diff = (np.array(ret[col].tolist()) - target[col]) / target[col]
            self.assertTrue(np.all(np.abs(diff) < 1e-8))

    def test_get_average_observables_wl_fill_factor_limit(self):
        """Tests get_average_observables_wl function with fill factor limit."""

        fill_factor_limit = (self.fill_factor_history[20] + self.fill_factor_history[30]) / 2

        # generate target data
        temperatures = [500, 1000, 1500, 3000, 5000]
        dc = self.prepare_dc_fill_factor_limit_ref()
        target = get_average_observables_wl(dc, temperatures, observables=['obs'])

        # test single container without observables
        dc = self.prepareDataContainer()
        ret = get_average_observables_wl(dc, temperatures, fill_factor_limit=fill_factor_limit)
        np.set_printoptions(precision=12)
        self.assertIsInstance(ret, DataFrame)
        for key in ['temperature', 'potential_mean', 'potential_std']:
            self.assertIn(key, ret.columns)
        for col in ret.columns:
            diff = (np.array(ret[col].tolist()) - target[col]) / target[col]
            self.assertTrue(np.all(np.abs(diff) < 1e-6))

        # test single container with observable
        dc = self.prepareDataContainer()
        ret = get_average_observables_wl(dc, temperatures, observables=['obs'],
                                         fill_factor_limit=fill_factor_limit)
        np.set_printoptions(precision=12)
        self.assertIsInstance(ret, DataFrame)
        for key in ['temperature', 'potential_mean', 'potential_std', 'obs_mean', 'obs_std']:
            self.assertIn(key, ret.columns)
        for col in ret.columns:
            diff = (np.array(ret[col].tolist()) - target[col]) / target[col]
            self.assertTrue(np.all(np.abs(diff) < 1e-6))

        # run with multiple containers
        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=4),
               2: self.prepareDataContainer(energy_limit_left=-4, energy_limit_right=None)}
        ret = get_average_observables_wl(dcs, temperatures, observables=['obs'],
                                         fill_factor_limit=fill_factor_limit)
        self.assertIsInstance(ret, DataFrame)
        for key in ['temperature', 'potential_mean', 'potential_std', 'obs_mean', 'obs_std']:
            self.assertIn(key, ret.columns)
        for col in ret.columns:
            diff = (np.array(ret[col].tolist()) - target[col]) / target[col]
            self.assertTrue(np.all(np.abs(diff) < 1e-8))

    def test_get_average_cluster_vectors_wl(self):
        """Tests get_average_observables_wl function."""

        cluster_space = ClusterSpace(self.prim, cutoffs=[1.1, 1.1], chemical_symbols=['Ag', 'Au'])
        temperatures = np.linspace(500, 1900, 4)
        target = {}
        target['cv_mean'] = [[1.0, -0.1402341191696221, 0.7963647151972209],
                             [1.0, -0.09959735947606872, 0.7880104645401809],
                             [1.0, -0.08138740512056972, 0.7840356680283108],
                             [1.0, -0.06898593935521304, 0.7813529484058864]]
        target['cv_std'] = [[0.0, 0.6203428368415204, 0.06008513126392547],
                            [0.0, 0.5990951919641202, 0.061738970036107135],
                            [0.0, 0.5864565057880073, 0.06201788853741115],
                            [0.0, 0.5768440885484074, 0.06196866651417745]]

        # test type warning
        with self.assertRaises(TypeError) as context:
            ret = get_average_cluster_vectors_wl('abc', cluster_space, temperatures)
        self.assertTrue('must be a data container with entropy data' in str(context.exception))

        # test single container without observables
        dc = self.prepareDataContainer()
        ret = get_average_cluster_vectors_wl(dc, cluster_space, temperatures)
        self.assertIsInstance(ret, DataFrame)
        for key in target:
            for row_ret, row_target in zip(ret[key].tolist(), target[key]):
                self.assertTrue(np.all(np.abs(np.array(row_ret) - row_target) < 1e-8))

        # test with multiple containers
        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=4),
               2: self.prepareDataContainer(energy_limit_left=-4, energy_limit_right=None)}
        ret = get_average_cluster_vectors_wl(dcs, cluster_space, temperatures)
        self.assertIsInstance(ret, DataFrame)
        for key in target:
            for row_ret, row_target in zip(ret[key].tolist(), target[key]):
                self.assertTrue(np.all(np.abs(np.array(row_ret) - row_target) < 1e-8))

    def test_get_average_cluster_vectors_wl_fill_factor_limit(self):
        """Tests get_average_observables_wl function with fill factor limit."""

        fill_factor_limit = (self.fill_factor_history[20] + self.fill_factor_history[30]) / 2

        # generate target data
        cluster_space = ClusterSpace(self.prim, cutoffs=[1.1, 1.1], chemical_symbols=['Ag', 'Au'])
        temperatures = np.linspace(500, 1900, 4)
        dc = self.prepare_dc_fill_factor_limit_ref()
        target = get_average_cluster_vectors_wl(dc, cluster_space, temperatures)

        # test single container without observables
        dc = self.prepareDataContainer()
        ret = get_average_cluster_vectors_wl(dc, cluster_space, temperatures,
                                             fill_factor_limit=fill_factor_limit)
        self.assertIsInstance(ret, DataFrame)
        for key in target:
            for row_ret, row_target in zip(ret[key].tolist(), target[key]):
                self.assertTrue(np.all(np.abs(np.array(row_ret) - row_target) < 1e-8))

        # test with multiple containers
        dcs = {1: self.prepareDataContainer(energy_limit_left=None, energy_limit_right=4),
               2: self.prepareDataContainer(energy_limit_left=-4, energy_limit_right=None)}
        ret = get_average_cluster_vectors_wl(dcs, cluster_space, temperatures,
                                             fill_factor_limit=fill_factor_limit)
        self.assertIsInstance(ret, DataFrame)
        for key in target:
            for row_ret, row_target in zip(ret[key].tolist(), target[key]):
                self.assertTrue(np.all(np.abs(np.array(row_ret) - row_target) < 1e-8))


if __name__ == '__main__':
    unittest.main()
