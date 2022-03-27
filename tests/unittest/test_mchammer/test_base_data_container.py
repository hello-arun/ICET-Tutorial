import os
import unittest
import tempfile
import random
import numpy as np
import pandas as pd
from ase.build import bulk
from collections import OrderedDict, Counter
from mchammer.data_containers.base_data_container import BaseDataContainer
from mchammer.observers.base_observer import BaseObserver


class ConcreteObserver(BaseObserver):
    """Child class of BaseObserver created for testing."""
    def __init__(self, interval, tag='ConcreteObserver'):
        super().__init__(interval=interval, return_type=int, tag=tag)

    def get_observable(self, structure):
        """Returns number of Al atoms."""
        return structure.get_chemical_symbols().count('Al')


class TestBaseDataContainer(unittest.TestCase):
    """Container for the tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestBaseDataContainer, self).__init__(*args, **kwargs)
        self.structure = bulk('Al').repeat(2)
        self.ensemble_parameters = {'number_of_atoms': len(self.structure), 'temperature': 375.15}

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test case."""
        self.dc = BaseDataContainer(structure=self.structure,
                                    ensemble_parameters=self.ensemble_parameters,
                                    metadata=OrderedDict(ensemble_name='test-ensemble', seed=144))

    def test_init(self):
        """Tests initializing BaseDataContainer."""
        self.assertIsInstance(self.dc, BaseDataContainer)

        # test fails with a non ASE Atoms type
        with self.assertRaises(TypeError) as context:
            BaseDataContainer(structure='structure',
                              ensemble_parameters=self.ensemble_parameters,
                              metadata=OrderedDict(ensemble_name='test-ensemble', seed=144))

        self.assertTrue('structure is not an ASE Atoms object' in str(context.exception))

    def test_structure(self):
        """Tests reference structure property."""
        self.assertEqual(self.dc.structure, self.structure)

    def test_append_data(self):
        """Tests append data functionality."""
        observers = [ConcreteObserver(interval=10, tag='obs1'),
                     ConcreteObserver(interval=20, tag='obs2')]

        mctrials = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        trajectory_write_interval = 50

        # append data
        observal_interval = min([obs.interval for obs in observers])
        for mctrial in mctrials:
            row_data = {}
            if mctrial % observal_interval == 0:
                for obs in observers:
                    if mctrial % obs.interval == 0:
                        observable = obs.get_observable(self.structure)
                        row_data[obs.tag] = observable
            if mctrial % trajectory_write_interval == 0:
                row_data['occupations'] = [13, 13, 13]
            self.dc.append(mctrial, row_data)

        # check number of entries
        counts = Counter([k for row in self.dc._data_list for k in row.keys()])
        self.assertEqual(counts['mctrial'], 10)
        self.assertEqual(counts['obs2'], 5)
        self.assertEqual(counts['occupations'], 2)

        # test whether method raises correct Exceptions
        with self.assertRaises(TypeError) as context:
            self.dc.append(5.0, 1.0)
        self.assertTrue('mctrial has the wrong type' in str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.dc.append(10, 1.0)
        self.assertTrue('mctrial values should be given in ascending order'
                        in str(context.exception))

        with self.assertRaises(TypeError) as context:
            self.dc.append(110, 'tst')
        self.assertTrue('record has the wrong type' in str(context.exception))

    def test_str(self):
        """Tests __str__ method."""
        data_rows = OrderedDict([
            (0, {'potential': -1.32,
                 'occupations': [14, 14, 14, 14, 14, 14, 14, 14]}),
            (10, {'potential': -1.35}),
            (20, {'potential': -1.33,
                  'occupations': [14, 13, 14, 14, 14, 14, 14, 14]}),
            (30, {'potential': -1.07}),
            (40, {'potential': -1.02,
                  'occupations': [14, 13, 13, 14, 14, 13, 14, 14]}),
            (50, {'potential': -1.4}),
            (60, {'potential': -1.3,
                  'occupations': [13, 13, 13, 13, 13, 13, 13, 14]})])
        for mctrial in data_rows:
            self.dc.append(mctrial, data_rows[mctrial])
        ret = str(self.dc)
        self.assertIn('n_rows_in_data', ret)
        self.assertIn('icet_version', ret)
        self.assertIn('data_container_type', ret)
        self.assertIn('BaseDataContainer', ret)

    def test_update_last_state(self):
        """Tests update_last_state functionality."""
        self.dc._update_last_state(last_step=10001,
                                   occupations=[13] * len(self.structure),
                                   accepted_trials=12,
                                   random_state=random.getstate())

        for key, value in self.dc._last_state.items():
            if key == 'last_step':
                self.assertIsInstance(value, int)
            if key == 'occupations':
                self.assertIsInstance(value, list)
            if key == 'accepted_trials':
                self.assertIsInstance(value, int)
            if key == 'random_state':
                self.assertIsInstance(value, tuple)

    def test_apply_observer(self):
        """ Tests apply observer """

        # generate dc with data and occupations
        data_rows = {0: {'potential': -1.32, 'occupations': [14, 14, 14, 14, 14, 14, 14, 14]},
                     10: {'potential': -1.35},
                     20: {'potential': -1.33, 'occupations': [14, 13, 14, 14, 14, 14, 14, 14]},
                     30: {'potential': -1.07},
                     40: {'potential': -1.02, 'occupations': [14, 13, 13, 14, 14, 13, 14, 14]},
                     50: {'potential': -1.4},
                     60: {'potential': -1.3, 'occupations': [13, 13, 13, 13, 13, 13, 13, 14]}}
        for mctrial in data_rows:
            self.dc.append(mctrial, data_rows[mctrial])

        # run new observer on
        class MyObserver(BaseObserver):
            def get_observable(self, structure):
                Al_count = structure.numbers.tolist().count(13)
                return Al_count**2

        new_observer = MyObserver(interval=1, return_type=float, tag='myobs')
        self.dc.apply_observer(new_observer)

        for row in self.dc._data_list:
            if 'occupations' in row:
                self.assertIn('myobs', row)
                target_obs = row['occupations'].count(13)**2
                self.assertEqual(target_obs, row['myobs'])

    def test_property_data(self):
        """Tests data property."""
        self.assertIsInstance(self.dc.data, pd.DataFrame)

    def test_property_parameters(self):
        """Tests parameters property."""
        self.assertEqual(self.dc.ensemble_parameters,
                         self.ensemble_parameters)

    def test_property_observables(self):
        """Tests observables property."""
        self.dc.append(10, {'obs1': 13,
                            'potential': -0.123})
        self.assertListEqual(sorted(self.dc.observables),
                             ['obs1', 'potential'])
        self.dc.append(20, {'obs2': 14})
        self.assertListEqual(sorted(self.dc.observables),
                             ['obs1', 'obs2', 'potential'])

    def test_property_metadata(self):
        """Tests get metadata method."""
        metadata = self.dc.metadata
        self.assertIn('seed', metadata.keys())
        self.assertIn('ensemble_name', metadata.keys())
        self.assertIn('username', metadata.keys())
        self.assertIn('hostname', metadata.keys())
        self.assertIn('icet_version', metadata.keys())

    def test_get(self):
        """ Tests the get function. """

        # dummy trajectory
        occ1 = [14, 14, 14, 14, 14, 14, 14, 14]
        occ2 = [13, 13, 14, 14, 14, 14, 14, 14]
        occ3 = [14, 14, 13, 13, 13, 13, 14, 14]
        traj = []
        for o in [occ1, occ2, occ3]:
            atoms = self.structure.copy()
            atoms.numbers = o
            traj.append(atoms)

        # append data to data container
        data_rows = OrderedDict([
            (0, {'obs1': 16, 'acceptance_ratio': 0.0, 'obs2': 11, 'occupations': occ1}),
            (10, {'acceptance_ratio': 0.9}),
            (20, {'obs1': 16, 'acceptance_ratio': 0.7}),
            (30, {'acceptance_ratio': 0.7, 'obs2': 13}),
            (40, {'obs1': 14, 'acceptance_ratio': 0.75}),
            (50, {'acceptance_ratio': 0.7, 'occupations': occ2}),
            (60, {'obs1': 16, 'acceptance_ratio': 0.6, 'obs2': 10}),
            (70, {'acceptance_ratio': 0.65}),
            (80, {'obs1': 14, 'acceptance_ratio': 0.66}),
            (90, {'acceptance_ratio': 0.666, 'obs2': 10}),
            (100, {'obs1': 16, 'acceptance_ratio': 0.7, 'occupations': occ3})])

        for mctrial in data_rows:
            self.dc.append(mctrial, data_rows[mctrial])

        # assert ValueError if no tags are given.
        with self.assertRaises(TypeError) as context:
            self.dc.get()
        self.assertIn('Missing tags argument', str(context.exception))

        # assert ValueError if bad tags are given.
        with self.assertRaises(ValueError) as context:
            self.dc.get('potential', 'asdqwerty')
        self.assertIn('No observable named', str(context.exception))

        # assert numpy array is returned
        mctrial, accept_ratio = self.dc.get('mctrial', 'acceptance_ratio')
        self.assertIsInstance(mctrial, np.ndarray)
        self.assertIsInstance(accept_ratio, np.ndarray)

        # with a given start
        mctrial, obs1 = self.dc.get('mctrial', 'obs1', start=60)
        self.assertEqual(mctrial.tolist(), [60, 80, 100])
        self.assertEqual(obs1.tolist(), [16, 14, 16])

        # get trajectory
        traj_ret = self.dc.get('trajectory')
        self.assertEqual(traj, traj_ret)

        # get trajectory with start
        traj_ret = self.dc.get('trajectory', start=30)
        self.assertEqual(traj[1], traj_ret[0])
        self.assertEqual(traj[2], traj_ret[1])

        # get trajectory with start and property
        traj_ret, obs1 = self.dc.get('trajectory', 'obs1', start=30)
        self.assertEqual(traj[2], traj_ret[0])
        self.assertEqual(obs1.tolist(), [16])

    def test_read_and_write(self):
        """Tests write and read functionalities of data container."""

        # append data for testing
        row_data = {}
        row_data['obs1'] = 64
        row_data['occupations'] = [13, 13, 13]
        for mctrial in range(1, 101):
            self.dc.append(mctrial, row_data)

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        # check before with a non-tar file
        with self.assertRaises(TypeError) as context:
            BaseDataContainer.read(temp_file.name)
        self.assertTrue('{} is not a tar file'.format(str(temp_file.name))
                        in str(context.exception))

        # save to file
        self.dc.write(temp_file.name)

        # read from file object
        dc_read = BaseDataContainer.read(temp_file)
        os.remove(temp_file.name)

        # check properties and metadata
        self.assertEqual(self.structure, dc_read.structure)
        self.assertEqual(self.dc.metadata, dc_read.metadata)
        self.assertEqual(self.dc.ensemble_parameters,
                         dc_read.ensemble_parameters)

        # check data
        pd.testing.assert_frame_equal(
            self.dc.data, dc_read.data, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
