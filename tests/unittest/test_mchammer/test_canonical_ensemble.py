import unittest

import numpy as np
from ase.build import bulk

from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles.canonical_ensemble import CanonicalEnsemble


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
        self.ensemble = CanonicalEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40,
            temperature=self.temperature)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(TypeError) as context:
            CanonicalEnsemble(structure=self.structure, calculator=self.calculator)
        self.assertTrue("required positional argument: 'temperature'" in
                        str(context.exception))

    def test_do_trial_step(self):
        """Tests the do trial step."""

        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()

    def test_run(self):
        """Test that run function runs. """
        n = 50
        self.ensemble.run(n)
        self.assertEqual(self.ensemble.step, n)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""

        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_property_boltzmann(self):
        """Tests init with explicit Boltzmann constant."""
        from ase.units import kB
        ens = CanonicalEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            user_tag='test-ensemble',
            random_seed=42, temperature=100.0)
        self.assertAlmostEqual(kB, ens.boltzmann_constant)

        ens = CanonicalEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            user_tag='test-ensemble',
            random_seed=42, temperature=100.0, boltzmann_constant=1.0)
        self.assertAlmostEqual(1.0, ens.boltzmann_constant)

    def test_property_temperature(self):
        """Tests property temperature."""
        self.assertEqual(self.ensemble.temperature, self.temperature)

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()
        self.assertIn('potential', data.keys())

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""

        n_atoms = len(self.structure)
        n_atoms_Al = self.structure.get_chemical_symbols().count('Al')
        n_atoms_Ga = self.structure.get_chemical_symbols().count('Ga')

        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'], n_atoms)
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms_Al'], n_atoms_Al)
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms_Ga'], n_atoms_Ga)
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'], self.temperature)

        # check in ensemble parameters was correctly passed to datacontainer
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'], n_atoms)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms_Al'], n_atoms_Al)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms_Ga'], n_atoms_Ga)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['temperature'], self.temperature)

    def test_write_interval_and_period(self):
        """Tests interval and period for writing data from ensemble."""
        self.assertEqual(self.ensemble._data_container_write_period, 499.0)
        self.assertEqual(self.ensemble._ensemble_data_write_interval, 25)
        self.assertEqual(self.ensemble._trajectory_write_interval, 40)

    def test_mc_with_one_filled_sublattice(self):
        """ Tests if canonical ensemble works with two sublattices
        where one sublattice is filled/empty. """

        # setup two sublattices
        prim = bulk('W', 'bcc', a=3.0, cubic=True)
        cs = ClusterSpace(prim, [4.0], [['W', 'Ti'], ['C', 'Be']])
        ce = ClusterExpansion(cs, np.arange(0, len(cs)))

        # setup supercell with one filled sublattice
        supercell = prim.copy()
        supercell[1].symbol = 'C'
        supercell = supercell.repeat(4)
        supercell[2].symbol = 'Ti'

        # run mc
        calc = ClusterExpansionCalculator(supercell, ce)

        mc = CanonicalEnsemble(supercell, calc, 300)
        mc.run(50)

    def test_get_swap_sublattice_probabilities(self):
        """ Tests the get_swap_sublattice_probabilities function. """

        # setup system with inactive sublattice
        prim = bulk('Al').repeat([2, 1, 1])
        chemical_symbols = [['Al'], ['Ag', 'Al']]
        cs = ClusterSpace(prim, cutoffs=[0], chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [1] * len(cs))

        supercell = prim.repeat(2)
        supercell[1].symbol = 'Ag'
        ce_calc = ClusterExpansionCalculator(supercell, ce)
        ensemble = CanonicalEnsemble(supercell, ce_calc, temperature=100)

        # test get_swap_sublattice_probabilities
        probs = ensemble._get_swap_sublattice_probabilities()
        self.assertEqual(len(probs), 2)
        self.assertEqual(probs[0], 1)
        self.assertEqual(probs[1], 0)

        # test raise when swap not possible on either lattice
        supercell[1].symbol = 'Al'
        ce_calc = ClusterExpansionCalculator(supercell, ce)

        with self.assertRaises(ValueError) as context:
            ensemble = CanonicalEnsemble(supercell, ce_calc, temperature=100)
        self.assertIn('No canonical swaps are possible on any of the', str(context.exception))


if __name__ == '__main__':
    unittest.main()
