import unittest
import numpy as np
from ase.build import bulk
from collections import OrderedDict
from mchammer.data_containers.data_container import DataContainer
from mchammer.observers.base_observer import BaseObserver
from mchammer.data_analysis import get_autocorrelation_function, get_correlation_length,\
    _estimate_correlation_length_from_acf, get_error_estimate, _estimate_error


class ConcreteObserver(BaseObserver):
    """Child class of BaseObserver created for testing."""
    def __init__(self, interval, tag='ConcreteObserver'):
        super().__init__(interval=interval, return_type=int, tag=tag)

    def get_observable(self, structure):
        """Returns number of Al atoms."""
        return structure.get_chemical_symbols().count('Al')


class TestDataContainer(unittest.TestCase):
    """Container for the tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestDataContainer, self).__init__(*args, **kwargs)
        self.structure = bulk('Al').repeat(2)
        self.ensemble_parameters = {'number_of_atoms': len(self.structure),
                                    'temperature': 375.15}

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test case."""
        self.dc = DataContainer(structure=self.structure,
                                ensemble_parameters=self.ensemble_parameters,
                                metadata=OrderedDict(ensemble_name='test-ensemble', seed=144))

    def test_init(self):
        """Tests initializing DataContainer."""
        self.assertIsInstance(self.dc, DataContainer)

        # test fails with a non ASE Atoms type
        with self.assertRaises(TypeError) as context:
            DataContainer(structure='structure',
                          ensemble_parameters=self.ensemble_parameters,
                          metadata=OrderedDict(ensemble_name='test-ensemble', seed=144))

        self.assertTrue('structure is not an ASE Atoms object' in str(context.exception))

    def test_analyze_data(self):
        """Tests analyze_data functionality."""

        # set up a random list of values with a normal distribution
        n_iter, mu, sigma = 100, 1.0, 0.1
        np.random.seed(12)
        for mctrial in range(n_iter):
            row = {'obs1': np.random.normal(mu, sigma), 'obs2': 4.0}
            self.dc.append(mctrial, record=row)

        # check obs1
        summary1 = self.dc.analyze_data('obs1')
        mean1 = self.dc.get('obs1').mean()
        std1 = self.dc.get('obs1').std()
        self.assertEqual(summary1['mean'], mean1)
        self.assertEqual(summary1['std'], std1)
        self.assertEqual(summary1['correlation_length'], 1)

        # check obs2
        summary2 = self.dc.analyze_data('obs2')
        self.assertTrue(np.isnan(summary2['correlation_length']))

    def test_get_average(self):
        """Tests get average functionality."""
        # set up a random list of values with a normal distribution
        n_iter, mu, sigma = 100, 1.0, 0.1
        np.random.seed(12)
        obs_val = np.random.normal(mu, sigma, n_iter).tolist()

        # append above random data to data container
        for mctrial in range(n_iter):
            self.dc.append(mctrial, record={'obs1': obs_val[mctrial]})

        # get average over all mctrials
        mean = self.dc.get_average('obs1')
        self.assertAlmostEqual(mean, 0.9855693, places=7)

        # get average over slice of data
        mean = self.dc.get_average('obs1', start=60)
        self.assertAlmostEqual(mean, 0.9851106, places=7)

        # test fails for non-existing data
        with self.assertRaises(ValueError) as context:
            self.dc.get_average('temperature')
        self.assertTrue('No observable named temperature' in str(context.exception))

        # test fails for non-scalar data
        with self.assertRaises(ValueError) as context:
            self.dc.get_average('trajectory')
        self.assertTrue('trajectory is not scalar' in str(context.exception))


class TestDataAnalysis(unittest.TestCase):
    """Container for the tests of the data_analysis functions. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.arange(1000) % 20

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_correlation_length(self):
        """ Tests get_correlation_length function. """

        acf = get_autocorrelation_function(self.data)

        # Correlation length test
        corr_len_target = 4
        corr_len1 = _estimate_correlation_length_from_acf(acf)
        corr_len2 = get_correlation_length(self.data)

        self.assertEqual(corr_len1, corr_len_target)
        self.assertEqual(corr_len2, corr_len_target)

        # Error estimate
        error_95_target = 0.7156495464548879
        error_95_1 = get_error_estimate(self.data)
        error_95_2 = _estimate_error(self.data, corr_len_target, 0.95)

        np.testing.assert_almost_equal(error_95_1, error_95_target)
        np.testing.assert_almost_equal(error_95_2, error_95_target)


if __name__ == '__main__':
    unittest.main()
