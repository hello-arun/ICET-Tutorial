import os
import unittest

import numpy as np
from ase import Atoms
from ase.build import bulk

from icet import ClusterExpansion, ClusterSpace
from icet.input_output.logging_tools import set_log_config
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import WangLandauEnsemble
from mchammer.observers.base_observer import BaseObserver

set_log_config(level=100)


class ConcreteObserver(BaseObserver):
    """Child class of BaseObserver created for testing."""
    def __init__(self, interval, tag='ConcreteObserver'):
        super().__init__(interval=interval, return_type=int, tag=tag)

    def get_observable(self, structure):
        """Returns number of Au atoms."""
        return structure.get_chemical_symbols().count('Au')


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        # prepare cluster expansion
        self.prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
        cs = ClusterSpace(self.prim, cutoffs=[1.1], chemical_symbols=['Ag', 'Au'])
        self.ce = ClusterExpansion(cs, [0, 0, 2])

        # prepare initial configuration
        self.structure = self.prim.repeat((2, 2, 1))
        self.structure[0].symbol = 'Ag'
        self.structure[1].symbol = 'Ag'
        self.structure = self.structure.repeat((2, 2, 1))

        # other variables and parameters
        self.trial_move = 'swap'
        self.energy_spacing = 1
        self.flatness_limit = 0.8
        self.fill_factor_limit = 1e-4
        self.flatness_check_interval = 10
        self.data_container_write_period = 499
        self.ensemble_data_write_interval = 1
        self.trajectory_write_interval = 10

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.structure, self.ce)
        self.ensemble = WangLandauEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            user_tag='test-ensemble',
            random_seed=42,
            trial_move=self.trial_move,
            energy_spacing=self.energy_spacing,
            flatness_limit=self.flatness_limit,
            flatness_check_interval=self.flatness_check_interval,
            fill_factor_limit=self.fill_factor_limit,
            data_container_write_period=self.data_container_write_period,
            ensemble_data_write_interval=self.ensemble_data_write_interval,
            trajectory_write_interval=self.trajectory_write_interval)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(TypeError) as context:
            WangLandauEnsemble(structure=self.structure, calculator=self.calculator)
        self.assertTrue("required positional argument: 'energy_spacing'" in
                        str(context.exception))

        # Try non-sensible trial move
        with self.assertRaises(ValueError) as context:
            WangLandauEnsemble(structure=self.structure,
                               calculator=self.calculator,
                               energy_limit_left=-2,
                               energy_limit_right=-10,
                               energy_spacing=self.energy_spacing,
                               random_seed=42)
        self.assertIn('Invalid energy window', str(context.exception))

        # Try non-sensible energy range
        with self.assertRaises(ValueError) as context:
            WangLandauEnsemble(structure=self.structure,
                               calculator=self.calculator,
                               trial_move='bizarre',
                               energy_spacing=self.energy_spacing,
                               random_seed=42)
        self.assertIn('Invalid value for trial_move', str(context.exception))

    def test_do_trial_step(self):
        """Tests trial steps."""

        # Use swap trial moves
        ens = WangLandauEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            trial_move='swap',
            energy_spacing=self.energy_spacing,
            random_seed=42)
        # By repeating the call several times one should generate both a
        # reject and an accept
        for _ in range(10):
            ens._do_trial_step()

        # Use flip trial moves
        ens = WangLandauEnsemble(
            structure=self.structure,
            calculator=self.calculator,
            trial_move='flip',
            energy_spacing=self.energy_spacing,
            random_seed=42)
        # By repeating the call several times one should generate both a
        # reject and an accept
        for _ in range(30):
            ens._do_trial_step()

    def test_run(self):
        """Test that run method runs. """
        n = 10
        self.ensemble.run(n)
        self.assertEqual(self.ensemble.step, n)

    def test_run_terminates(self):
        """Test that a run terminates if convergence is reached. """

        # ensemble
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 flatness_limit=0.0,
                                 flatness_check_interval=1,
                                 fill_factor_limit=0.25,
                                 energy_spacing=self.energy_spacing,
                                 ensemble_data_write_interval=1)

        ens.run(10)
        self.assertTrue(ens._converged)
        self.assertEqual(ens._fill_factor, 0.25)
        self.assertEqual(ens._step, 4)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""
        # since the outcome is non-trivial run for both positive and negative
        # potential difference without expectation of a specific outcome
        self.ensemble._acceptance_condition(-1.0)
        self.ensemble._acceptance_condition(10000.0)

        # test rejection due to unallowed move by shrinking the energy window
        # - prepare initial configuration with energy 0
        structure = self.prim.repeat((2, 2, 1))
        structure[0].symbol = 'Ag'
        structure[1].symbol = 'Ag'
        structure = structure.repeat((2, 2, 1))
        ens = WangLandauEnsemble(structure, self.calculator, energy_spacing=1,
                                 energy_limit_left=0,
                                 energy_limit_right=1,
                                 ensemble_data_write_interval=1,
                                 random_seed=42)
        ens.run(10)

        # test
        # - prepare initial configuration with energy -32 (ground state)
        structure = self.prim.repeat((2, 2, 1))
        structure[0].symbol = 'Ag'
        structure[3].symbol = 'Ag'
        structure = structure.repeat((2, 2, 1))
        ens = WangLandauEnsemble(structure, self.calculator, energy_spacing=1,
                                 energy_limit_left=-16,
                                 ensemble_data_write_interval=1,
                                 random_seed=42)
        ens.run(10)

        # use unreachable energy window to test window approach aspects
        # - prepare initial configuration with energy -32 (ground state)
        structure = self.prim.repeat((2, 2, 1))
        structure[0].symbol = 'Ag'
        structure[3].symbol = 'Ag'
        structure = structure.repeat((2, 2, 1))
        ens = WangLandauEnsemble(structure, self.calculator, energy_spacing=1,
                                 energy_limit_right=-60,
                                 energy_limit_left=-70,
                                 ensemble_data_write_interval=1,
                                 random_seed=42)
        ens.run(10)  # Run to get something in the data container

        # Approaching the window should always be allowed
        self.assertTrue(ens._acceptance_condition(-10))

        # Stepping far away from window should not be allowed
        self.assertFalse(ens._acceptance_condition(10000.0))

        # Do the same thing from below
        ens._potential = -100
        self.assertTrue(ens._acceptance_condition(10))
        self.assertFalse(ens._acceptance_condition(-10000.0))

        # Take a step that would be rejected due to entropy but accepted
        # because it takes us closer to window
        self.assertEqual(ens._potential, -90)
        ens._histogram[-80] = 1e9
        ens._entropy[-80] = 1e9
        self.assertTrue(ens._acceptance_condition(10))

        # Take a step that would be accepted due to entropy but rejected
        # because it takes us away to window
        self.assertEqual(ens._potential, -80)
        ens._histogram[-80] = 100
        ens._entropy[-80] = 100
        ens._histogram[-10000] = 1
        ens._entropy[-10000] = 1
        self.assertFalse(ens._acceptance_condition(-9920))

        # Stepping away should not be allowed if the penalty is high
        ens._potential = -80
        ens._histogram[-81] = 1
        ens._entropy[-81] = 1
        ens._window_search_penalty = 100
        self.assertFalse(ens._acceptance_condition(-1))

        # Stepping away should be allowed if the penalty is low
        ens._window_search_penalty = 0.0001
        self.assertTrue(ens._acceptance_condition(-1))

        # Finally step inside the window
        ens._potential = -59
        self.assertTrue(ens._acceptance_condition(-5))
        self.assertTrue(ens._reached_energy_window)

        # Todo: come up with more sensitive tests

    def test_property_fill_factor(self):
        """Tests behavior of flatness_limit."""
        self.assertAlmostEqual(1, self.ensemble.fill_factor)

    def test_property_fill_factor_history(self):
        """Tests behavior of flatness_limit."""
        ret = self.ensemble.fill_factor_history
        target = {0: 1}
        self.assertDictEqual(ret, target)
        self.ensemble.flatness_limit = 0
        self.ensemble.flatness_check_interval = 2
        self.ensemble.run(10)
        ret = self.ensemble.fill_factor_history
        target = {0: 1, 2: 0.5, 4: 0.25, 6: 0.125, 8: 0.0625}
        self.assertDictEqual(ret, target)

    def test_property_converged(self):
        """Tests behavior of converged property."""
        self.assertIsNone(self.ensemble.converged)
        self.ensemble.flatness_check_interval = 2
        self.ensemble.run(4)
        self.assertFalse(self.ensemble.converged)
        self.ensemble.fill_factor_limit = 2
        self.ensemble.flatness_limit = 0
        self.ensemble.run(4)
        self.assertTrue(self.ensemble.converged)

    def test_property_fill_factor_limit(self):
        """Tests behavior of fill_factor_limit property."""
        self.assertAlmostEqual(self.fill_factor_limit, self.ensemble.fill_factor_limit)
        self.ensemble.fill_factor_limit = 0.02
        self.assertAlmostEqual(0.02, self.ensemble.fill_factor_limit)

    def test_property_flatness_limit(self):
        """Tests behavior of flatness_limit property."""
        self.assertAlmostEqual(self.flatness_limit, self.ensemble.flatness_limit)
        self.ensemble.flatness_limit = 0.7
        self.assertAlmostEqual(0.7, self.ensemble.flatness_limit)

    def test_property_flatness_check_interval(self):
        """Tests behavior of flatness_check_interval  property."""
        self.assertEqual(10, self.ensemble.flatness_check_interval)
        self.ensemble.flatness_check_interval = 200
        self.assertAlmostEqual(200, self.ensemble.flatness_check_interval)

    def test_get_ensemble_data(self):
        """Tests the get_ensemble_data method."""
        data = self.ensemble._get_ensemble_data()
        self.assertIn('potential', data.keys())

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""

        n_atoms = len(self.structure)
        n_atoms_Au = self.structure.get_chemical_symbols().count('Au')
        n_atoms_Ag = self.structure.get_chemical_symbols().count('Ag')

        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'], n_atoms)
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms_Au'], n_atoms_Au)
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms_Ag'], n_atoms_Ag)
        self.assertEqual(self.ensemble.ensemble_parameters['energy_spacing'], self.energy_spacing)
        self.assertEqual(self.ensemble.ensemble_parameters['trial_move'], self.trial_move)

        # check in ensemble parameters was correctly passed to datacontainer
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms'], n_atoms)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms_Au'], n_atoms_Au)
        self.assertEqual(self.ensemble.data_container.ensemble_parameters['n_atoms_Ag'], n_atoms_Ag)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['energy_spacing'], self.energy_spacing)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['trial_move'], self.trial_move)

    def test_write_interval_and_period(self):
        """Tests interval and period for writing data from ensemble."""
        self.assertEqual(self.ensemble._data_container_write_period,
                         self.data_container_write_period)
        self.assertEqual(self.ensemble._ensemble_data_write_interval,
                         self.ensemble_data_write_interval)
        self.assertEqual(self.ensemble._trajectory_write_interval,
                         self.trajectory_write_interval)

    def test_entropy_history(self):
        """ Tests if the entropy history is updated """
        self.assertTrue(len(self.ensemble._entropy_history)
                        == len(self.ensemble._fill_factor_history) - 1)
        self.ensemble.flatness_check_interval = 2
        self.ensemble.run(4)
        self.assertTrue(len(self.ensemble._entropy_history)
                        == len(self.ensemble._fill_factor_history) - 1)
        self.ensemble.flatness_limit = 0
        self.ensemble.run(4)
        self.assertTrue(len(self.ensemble._entropy_history)
                        == len(self.ensemble._fill_factor_history) - 1)
        for target_mctrial, ret_mctrial in zip(list(self.ensemble._fill_factor_history.keys())[1:],
                                               self.ensemble._entropy_history.keys()):
            self.assertEqual(target_mctrial, ret_mctrial)

    def test_mc_with_one_filled_sublattice(self):
        """ Tests if WL simulation works with two sublattices
        where one sublattice is filled/empty. """

        # setup two sublattices
        prim = bulk('W', crystalstructure='bcc', a=1.0, cubic=True)
        cs = ClusterSpace(prim, [1.5], [['W', 'Ti'], ['C', 'Be']])
        ce = ClusterExpansion(cs, [1] * len(cs))

        # setup supercell with one filled sublattice
        structure = prim.copy()
        structure[1].symbol = 'C'
        structure = structure.repeat(4)
        structure[2].symbol = 'Ti'

        # run mc
        calculator = ClusterExpansionCalculator(structure, ce)
        mc = WangLandauEnsemble(structure, calculator, energy_spacing=1)
        mc.run(50)

    def test_get_sublattice_probabilities(self):
        """ Tests the get_swap/flip_sublattice_probabilities function. """

        # setup system with inactive sublattice
        prim = bulk('Al').repeat([2, 1, 1])
        chemical_symbols = [['Al'], ['Ag', 'Al']]
        cs = ClusterSpace(prim, cutoffs=[0], chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [1] * len(cs))

        structure = prim.repeat(2)
        structure[1].symbol = 'Ag'
        calculator = ClusterExpansionCalculator(structure, ce)
        ensemble = WangLandauEnsemble(structure, calculator, energy_spacing=1)

        # test get_swap_sublattice_probabilities
        probs = ensemble._get_swap_sublattice_probabilities()
        self.assertEqual(len(probs), 2)
        self.assertEqual(probs[0], 1)
        self.assertEqual(probs[1], 0)

        # test get_flip_sublattice_probabilities
        probs = ensemble._get_flip_sublattice_probabilities()
        self.assertEqual(len(probs), 2)
        self.assertEqual(probs[0], 1)
        self.assertEqual(probs[1], 0)

        # test raise when swap not possible on either lattice
        structure[1].symbol = 'Al'
        calculator = ClusterExpansionCalculator(structure, ce)

        with self.assertRaises(ValueError) as context:
            ensemble = WangLandauEnsemble(structure, calculator, energy_spacing=1)
        self.assertIn('No swaps are possible on any of the', str(context.exception))

    def test_sublattice_probabilities(self):
        """ Tests the sublattice_probabilities keyword argument. """

        # setup system with inactive sublattice
        prim = bulk('W', 'bcc', cubic=True)
        prim[1].symbol = 'C'
        chemical_symbols = [['W', 'Ti'], ['C', 'N']]
        cs = ClusterSpace(prim, cutoffs=[0], chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [1] * len(cs))
        structure = prim.repeat(2)
        structure[0].symbol = 'Ti'
        structure[1].symbol = 'N'
        calculator = ClusterExpansionCalculator(structure, ce)

        # test default
        ensemble = WangLandauEnsemble(structure, calculator, energy_spacing=1)
        probs = ensemble._get_swap_sublattice_probabilities()
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.5)

        # test override
        ensemble = WangLandauEnsemble(structure, calculator, energy_spacing=1,
                                      sublattice_probabilities=[0.2, 0.8])
        probs = ensemble._sublattice_probabilities
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs[0], 0.2)
        self.assertAlmostEqual(probs[1], 0.8)

    def test_restart_ensemble(self):
        """ Tests the restart functionality. """
        dc_filename = 'my-test.dc'
        # ensemble for first run
        ens1 = WangLandauEnsemble(structure=self.structure,
                                  calculator=self.calculator,
                                  dc_filename=dc_filename,
                                  energy_spacing=self.energy_spacing,
                                  ensemble_data_write_interval=2)
        ens1.run(10)

        # ensemble for second run
        ens2 = WangLandauEnsemble(structure=self.structure,
                                  calculator=self.calculator,
                                  dc_filename=dc_filename,
                                  energy_spacing=self.energy_spacing,
                                  ensemble_data_write_interval=2)
        self.assertEqual(len(ens1.data_container.data), len(ens2.data_container.data))
        self.assertTrue(np.allclose(list(ens1.data_container.data.potential),
                                    list(ens2.data_container.data.potential)))

        # ensure that the simulation is not run if converged
        ens2._converged = True
        ens2.run(10)
        self.assertEqual(ens1.step, ens2.step)
        self.assertEqual(len(ens1.data_container.data), len(ens2.data_container.data))

        # ensure that the simulation is run if not converged
        ens2._converged = False
        ens2.run(10)
        self.assertEqual(ens1.step + 10, ens2.step)
        self.assertEqual(len(ens1.data_container.data) + 5, len(ens2.data_container.data))

        os.remove(dc_filename)

        # ensemble for third run
        ens3 = WangLandauEnsemble(structure=self.structure,
                                  calculator=self.calculator,
                                  dc_filename=dc_filename,
                                  flatness_limit=0.0,
                                  flatness_check_interval=1,
                                  fill_factor_limit=0.5 ** 3,
                                  energy_spacing=self.energy_spacing,
                                  ensemble_data_write_interval=1)

        ens3.run(2)
        self.assertFalse(ens3._converged)
        self.assertEqual(ens3._fill_factor, 0.5)
        ens3.run(4)
        self.assertTrue(ens3._converged)
        self.assertEqual(ens3._fill_factor, 0.125)

        # ensemble for fourth run
        ens4 = WangLandauEnsemble(structure=self.structure,
                                  calculator=self.calculator,
                                  dc_filename=dc_filename,
                                  flatness_limit=0.8,
                                  flatness_check_interval=1,
                                  fill_factor_limit=0.5 ** 3,
                                  energy_spacing=self.energy_spacing,
                                  ensemble_data_write_interval=1)
        ens4.run(4)
        self.assertEqual(ens3.step + 4, ens4.step)
        self.assertFalse(ens4._converged)

        # ensemble for fifth run
        ens5 = WangLandauEnsemble(structure=self.structure,
                                  calculator=self.calculator,
                                  dc_filename=dc_filename,
                                  flatness_limit=0.0,
                                  flatness_check_interval=1,
                                  fill_factor_limit=0.5 ** 20,
                                  energy_spacing=self.energy_spacing,
                                  ensemble_data_write_interval=1)
        ens5.run(5)
        self.assertEqual(ens4.step + 5, ens5.step)
        self.assertFalse(ens5._converged)

        os.remove(dc_filename)

    def test_allow_move(self):
        """ Tests the allow_move method. """
        # with no limit
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing)
        self.assertTrue(ens._allow_move(bin_cur=-1, bin_new=-10))
        self.assertTrue(ens._allow_move(bin_cur=1, bin_new=-1))
        self.assertTrue(ens._allow_move(bin_cur=-1, bin_new=1))
        self.assertTrue(ens._allow_move(bin_cur=1, bin_new=10))
        # with left limit
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_left=0)
        self.assertTrue(ens._allow_move(bin_cur=-1, bin_new=-10))
        self.assertFalse(ens._allow_move(bin_cur=1, bin_new=-1))
        self.assertTrue(ens._allow_move(bin_cur=-1, bin_new=1))
        self.assertTrue(ens._allow_move(bin_cur=1, bin_new=10))
        # with right limit
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_right=0)
        self.assertTrue(ens._allow_move(bin_cur=-1, bin_new=-10))
        self.assertTrue(ens._allow_move(bin_cur=1, bin_new=-1))
        self.assertFalse(ens._allow_move(bin_cur=-1, bin_new=1))
        self.assertTrue(ens._allow_move(bin_cur=1, bin_new=10))
        # with left and right limits
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_left=-2,
                                 energy_limit_right=2)
        self.assertFalse(ens._allow_move(bin_cur=-1, bin_new=-10))
        self.assertTrue(ens._allow_move(bin_cur=1, bin_new=-1))
        self.assertTrue(ens._allow_move(bin_cur=-1, bin_new=1))
        self.assertFalse(ens._allow_move(bin_cur=1, bin_new=10))

    def test_inside_energy_window(self):
        """ Tests the inside_energy_window method. """
        # with no limit
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing)
        self.assertTrue(ens._inside_energy_window(-1))
        self.assertTrue(ens._inside_energy_window(1))
        # with left limit
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_left=0)
        self.assertFalse(ens._inside_energy_window(-1))
        self.assertTrue(ens._inside_energy_window(1))
        # with right limit
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_right=0)
        self.assertTrue(ens._inside_energy_window(-1))
        self.assertFalse(ens._inside_energy_window(1))
        # with left and limits
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_left=1,
                                 energy_limit_right=4)
        self.assertFalse(ens._inside_energy_window(-1))
        self.assertTrue(ens._inside_energy_window(1))

    def test_reached_energy_window(self):
        """ Tests the reached_energy_window property. """
        # with no limit
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing)
        self.assertTrue(ens._reached_energy_window)
        # with left limit < energy of structure (0)
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_left=-1)
        self.assertTrue(ens._reached_energy_window)
        # with left limit > energy of structure (0)
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_left=1)
        self.assertFalse(ens._reached_energy_window)
        # with right limit < energy of structure (0)
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_right=-1)
        self.assertFalse(ens._reached_energy_window)
        # with right limit > energy of structure (0)
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_right=1)
        self.assertTrue(ens._reached_energy_window)
        # with energy of structure (0) within left and right limits
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_left=-1,
                                 energy_limit_right=1)
        self.assertTrue(ens._reached_energy_window)
        # with energy of structure (0) outside left and right limits
        ens = WangLandauEnsemble(structure=self.structure,
                                 calculator=self.calculator,
                                 energy_spacing=self.energy_spacing,
                                 energy_limit_left=1,
                                 energy_limit_right=2)
        self.assertFalse(ens._reached_energy_window)


if __name__ == '__main__':
    unittest.main()
