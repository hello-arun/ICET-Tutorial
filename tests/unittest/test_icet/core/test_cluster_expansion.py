import os
import sys
import tempfile
import unittest
from io import StringIO

from icet import ClusterSpace, ClusterExpansion
from ase.build import bulk
from ase import Atoms  # NOQA (needed for eval(retval))
import numpy as np


def strip_surrounding_spaces(input_string):
    """
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    """
    s = []
    for line in StringIO(input_string):
        if len(line.strip()) == 0:
            continue
        s += [line.strip()]
    return '\n'.join(s)


class TestClusterExpansion(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterExpansion, self).__init__(*args, **kwargs)
        self.structure = bulk('Au')
        self.cutoffs = [3.0] * 3
        chemical_symbols = ['Au', 'Pd']
        self.cs = ClusterSpace(self.structure, self.cutoffs, chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        params_len = len(self.cs)
        self.parameters = np.arange(params_len)
        self.ce = ClusterExpansion(self.cs, self.parameters)

    def test_init(self):
        """Tests that initialization works."""
        self.assertIsInstance(self.ce, ClusterExpansion)

        # test whether method raises Exception
        with self.assertRaises(ValueError) as context:
            ClusterExpansion(self.cs, [0.0])
        self.assertTrue('cluster_space (5) and parameters (1) must have the'
                        ' same length' in str(context.exception))

    def test_predict(self):
        """Tests predict function."""
        predicted_val = self.ce.predict(self.structure)
        self.assertEqual(predicted_val, 10.0)

    def test_property_orders(self):
        """Tests orders property."""
        self.assertEqual(self.ce.orders, list(range(len(self.cutoffs) + 2)))

    def test_property_to_dataframe(self):
        """Tests to_dataframe() property."""
        df = self.ce.to_dataframe()
        self.assertIn('radius', df.columns)
        self.assertIn('order', df.columns)
        self.assertIn('eci', df.columns)
        self.assertEqual(len(df), len(self.parameters))

    def test_get__clusterspace_copy(self):
        """Tests get cluster space copy."""
        self.assertEqual(str(self.ce.get_cluster_space_copy()), str(self.cs))

    def test_property_parameters(self):
        """Tests parameters properties."""
        self.assertEqual(list(self.ce.parameters), list(self.parameters))

    def test_len(self):
        """Tests len functionality."""
        self.assertEqual(self.ce.__len__(), len(self.parameters))

    def test_read_write(self):
        """Tests read and write functionalities."""
        # save to file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        self.ce.write(temp_file.name)

        # read from file
        ce_read = ClusterExpansion.read(temp_file.name)
        os.remove(temp_file.name)

        # check cluster space
        self.assertEqual(self.cs._input_structure, ce_read._cluster_space._input_structure)
        self.assertEqual(self.cs._cutoffs, ce_read._cluster_space._cutoffs)
        self.assertEqual(
            self.cs._input_chemical_symbols, ce_read._cluster_space._input_chemical_symbols)

        # check parameters
        self.assertIsInstance(ce_read.parameters, np.ndarray)
        self.assertEqual(list(ce_read.parameters), list(self.parameters))

        # check metadata
        self.assertEqual(len(self.ce.metadata), len(ce_read.metadata))
        self.assertSequenceEqual(sorted(self.ce.metadata.keys()), sorted(ce_read.metadata.keys()))
        for key in self.ce.metadata.keys():
            self.assertEqual(self.ce.metadata[key], ce_read.metadata[key])

    def test_read_write_pruned(self):
        """Tests read and write functionalities."""
        # save to file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.ce.prune(indices=[2, 3])
        self.ce.prune(tol=3)
        pruned_params = self.ce.parameters
        pruned_cs_len = len(self.ce._cluster_space)
        temp_file.close()
        self.ce.write(temp_file.name)

        # read from file
        ce_read = ClusterExpansion.read(temp_file.name)
        params_read = ce_read.parameters
        cs_len_read = len(ce_read._cluster_space)
        os.remove(temp_file.name)

        # check cluster space
        self.assertEqual(cs_len_read, pruned_cs_len)
        self.assertEqual(list(params_read), list(pruned_params))

    def test_prune_cluster_expansion(self):
        """Tests pruning cluster expansion."""
        len_before = len(self.ce)
        self.ce.prune()
        len_after = len(self.ce)
        self.assertEqual(len_before, len_after)

        # Set all parameters to zero except three
        self.ce._parameters = np.array([0.0] * len_after)
        self.ce._parameters[0] = 1.0
        self.ce._parameters[1] = 2.0
        self.ce._parameters[2] = 0.5
        self.ce.prune()
        self.assertEqual(len(self.ce), 3)
        self.assertNotEqual(len(self.ce), len_after)

    def test_prune_cluster_expansion_tol(self):
        """Tests pruning cluster expansion with tolerance."""
        len_before = len(self.ce)
        self.ce.prune()
        len_after = len(self.ce)
        self.assertEqual(len_before, len_after)

        # Set all parameters to zero except two, one of which is
        # non-zero but below the tolerance
        self.ce._parameters = np.array([0.0] * len_after)
        self.ce._parameters[0] = 1.0
        self.ce._parameters[1] = 0.01
        self.ce.prune(tol=0.02)
        self.assertEqual(len(self.ce), 1)
        self.assertNotEqual(len(self.ce), len_after)

    def test_prune_pairs(self):
        """Tests pruning pairs only."""
        df = self.ce.to_dataframe()
        pair_indices = df.index[df['order'] == 2].tolist()
        self.ce.prune(indices=pair_indices)

        df_new = self.ce.to_dataframe()
        pair_indices_new = df_new.index[df_new['order'] == 2].tolist()
        self.assertEqual(pair_indices_new, [])

    def test_prune_zerolet(self):
        """Tests pruning zerolet."""
        with self.assertRaises(ValueError) as context:
            self.ce.prune(indices=[0])
        self.assertTrue('zerolet may not be pruned' in str(context.exception))

    def test_repr(self):
        """Tests __repr__ method."""
        retval = self.ce.__repr__()
        self.assertIn('ClusterExpansion', retval)
        self.assertIn('ClusterSpace', retval)
        ret = eval(retval)
        self.assertIsInstance(ret, ClusterExpansion)

    def test_str(self):
        """Tests __str__ method."""
        retval = self.ce.__str__()
        target = """
================================================ Cluster Expansion ================================================
 space group                            : Fm-3m (225)
 chemical species                       : ['Au', 'Pd'] (sublattice A)
 cutoffs                                : 3.0000 3.0000 3.0000
 total number of parameters             : 5
 number of parameters by order          : 0= 1  1= 1  2= 1  3= 1  4= 1
 fractional_position_tolerance          : 2e-06
 position_tolerance                     : 1e-05
 symprec                                : 1e-05
 total number of nonzero parameters     : 4
 number of nonzero parameters by order  : 0= 0  1= 1  2= 1  3= 1  4= 1
-------------------------------------------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multicomponent_vector | sublattices | parameter |    ECI
-------------------------------------------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .           |      .      |         0 |         0
   1  |   1   |   0.0000 |        1     |       0     |          [0]          |      A      |         1 |         1
   2  |   2   |   1.4425 |        6     |       1     |        [0, 0]         |     A-A     |         2 |     0.333
   3  |   3   |   1.6657 |        8     |       2     |       [0, 0, 0]       |    A-A-A    |         3 |     0.375
   4  |   4   |   1.7667 |        2     |       3     |     [0, 0, 0, 0]      |   A-A-A-A   |         4 |         2
===================================================================================================================
"""  # noqa

        self.assertEqual(strip_surrounding_spaces(target), strip_surrounding_spaces(retval))

    def test_get_string_representation(self):
        """Tests _get_string_representation functionality."""
        retval = self.ce._get_string_representation(print_threshold=2, print_minimum=1)
        target = """
================================================ Cluster Expansion ================================================
 space group                            : Fm-3m (225)
 chemical species                       : ['Au', 'Pd'] (sublattice A)
 cutoffs                                : 3.0000 3.0000 3.0000
 total number of parameters             : 5
 number of parameters by order          : 0= 1  1= 1  2= 1  3= 1  4= 1
 fractional_position_tolerance          : 2e-06
 position_tolerance                     : 1e-05
 symprec                                : 1e-05
 total number of nonzero parameters     : 4
 number of nonzero parameters by order  : 0= 0  1= 1  2= 1  3= 1  4= 1
-------------------------------------------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multicomponent_vector | sublattices | parameter |    ECI
-------------------------------------------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .           |      .      |         0 |         0
 ...
   4  |   4   |   1.7667 |        2     |       3     |     [0, 0, 0, 0]      |   A-A-A-A   |         4 |         2
===================================================================================================================
"""  # noqa

        self.assertEqual(strip_surrounding_spaces(target), strip_surrounding_spaces(retval))

    def test_print_overview(self):
        """Tests print_overview functionality."""
        with StringIO() as capturedOutput:
            sys.stdout = capturedOutput  # redirect stdout
            self.ce.print_overview()
            sys.stdout = sys.__stdout__  # reset redirect
            self.assertTrue('Cluster Expansion' in capturedOutput.getvalue())


class TestClusterExpansionTernary(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterExpansionTernary, self).__init__(*args, **kwargs)
        self.structure = bulk('Au')
        self.cutoffs = [3.0] * 3
        chemical_symbols = ['Au', 'Pd', 'Ag']
        self.cs = ClusterSpace(self.structure, self.cutoffs, chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        params_len = len(self.cs)
        self.parameters = np.arange(params_len)
        self.ce = ClusterExpansion(self.cs, self.parameters)

    def test_prune_cluster_expansion_with_indices(self):
        """Tests pruning cluster expansion."""

        self.ce.prune(indices=[1, 2, 3, 4, 5])

    def test_prune_cluster_expansion_with_tol(self):
        """Tests pruning cluster expansion."""
        # Prune everything
        self.ce.prune(tol=1e3)
        self.assertEqual(len(self.ce), 1)

    def test_prune_pairs(self):
        """Tests pruning pairs only"""

        df = self.ce.to_dataframe()
        pair_indices = df.index[df['order'] == 2].tolist()
        self.ce.prune(indices=pair_indices)

        df_new = self.ce.to_dataframe()
        pair_indices_new = df_new.index[df_new['order'] == 2].tolist()
        self.assertEqual(pair_indices_new, [])

    def test_property_metadata(self):
        """ Test metadata property. """

        user_metadata = dict(parameters=[1, 2, 3], fit_method='ardr')
        ce = ClusterExpansion(self.cs, self.parameters, metadata=user_metadata)
        metadata = ce.metadata

        # check for user metadata
        self.assertIn('parameters', metadata.keys())
        self.assertIn('fit_method', metadata.keys())

        # check for default metadata
        self.assertIn('date_created', metadata.keys())
        self.assertIn('username', metadata.keys())
        self.assertIn('hostname', metadata.keys())
        self.assertIn('icet_version', metadata.keys())

    def test_property_primitive_structure(self):
        """ Test primitive_structure property.. """
        prim = self.cs.primitive_structure
        self.assertEqual(prim, self.ce.primitive_structure)


if __name__ == '__main__':
    unittest.main()
