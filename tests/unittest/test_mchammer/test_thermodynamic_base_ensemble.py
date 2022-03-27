import unittest
import numpy as np
from ase.build import bulk
from ase.data import atomic_numbers
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
from mchammer.ensembles.vcsgc_ensemble import get_phis


def _assertAlmostEqualDict(self, retval, target, places=6):
    """
    Helper function that conducts an element-wise comparison of a
    dictionary.
    """
    self.assertIsInstance(retval, type(target))
    for key, val in target.items():
        self.assertIn(key, retval)
        s = ["key: {}({})".format(key, type(key))]
        s += ["retval: {} ({})".format(retval[key], type(retval[key]))]
        s += ["target: {} ({})".format(val, type(val))]
        info = '   '.join(s)
        self.assertAlmostEqual(val, retval[key], places=places, msg=info)


unittest.TestCase.assertAlmostEqualDict = _assertAlmostEqualDict


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        self.structure = bulk('Al').repeat(3)
        for i, atom in enumerate(self.structure):
            if i % 2 == 0:
                atom.symbol = 'Ga'
        cutoffs = [5, 5, 4]
        self.elements = ['Al', 'Ga']
        self.chemical_potentials = {'Al': 5, 'Ga': 0}
        self.cs = ClusterSpace(self.structure, cutoffs, self.elements)
        parameters = parameters = np.array([1.2] * len(self.cs))
        self.ce = ClusterExpansion(self.cs, parameters)
        self.temperature = 100.0

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.structure, self.ce)

        self.structure = bulk('Al').repeat(3)
        for i, atom in enumerate(self.structure):
            if i % 2 == 0:
                atom.symbol = 'Ga'

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

    def test_do_sgc_trial_step(self):
        """Tests the do trial step."""
        chemical_potentials = self.ensemble._chemical_potentials

        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_sgc_flip(sublattice_index=sl_index,
                                      chemical_potentials=chemical_potentials)

        # repeat the test when specifying allowed species
        allowed_species = [atomic_numbers[s] for s in self.elements]
        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_sgc_flip(sublattice_index=sl_index,
                                      chemical_potentials=chemical_potentials,
                                      allowed_species=allowed_species)

    def test_do_canonical_trial_step(self):
        """Tests the do trial step."""

        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_canonical_swap(sublattice_index=sl_index)

        # repeat the test when specifying allowed species
        allowed_species = [atomic_numbers[s] for s in self.elements]
        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_canonical_swap(sublattice_index=sl_index,
                                            allowed_species=allowed_species)

    def test_do_vcsgc_flip(self):
        """Test the vcsgc flip."""
        kappa = 200
        phis = {'Al': -1}
        phis = get_phis(phis)
        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_vcsgc_flip(phis=phis, kappa=kappa, sublattice_index=sl_index)

        # repeat the test when specifying allowed species
        allowed_species = [atomic_numbers[s] for s in self.elements]
        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_vcsgc_flip(phis=phis, kappa=kappa, sublattice_index=sl_index,
                                        allowed_species=allowed_species)

    def test_get_vcsgc_free_energy_derivatives(self):
        """Test the functionality for determining the VCSGC free energy derivatives."""
        kappa = 200
        phis = {'Al': -1}
        phis = get_phis(phis)
        concentration = len(
            [n for n in self.structure.numbers if n == 13]) / len(self.structure)
        target = {'free_energy_derivative_Al': kappa * self.ensemble.boltzmann_constant *
                  self.temperature * (- 2 * concentration - phis[13])}
        data = self.ensemble._get_vcsgc_free_energy_derivatives(phis=phis, kappa=kappa)
        self.assertAlmostEqualDict(data, target)

        # repeat the test when specifying a sublattice index
        sublattice_index = 0
        data = self.ensemble._get_vcsgc_free_energy_derivatives(phis=phis, kappa=kappa,
                                                                sublattice_index=sublattice_index)
        self.assertAlmostEqualDict(data, target)

    def test_get_species_counts(self):
        """Test the functionality for determining the species counts."""
        target = {'{}_count'.format(symbol): counts for symbol, counts in
                  zip(*np.unique(self.structure.get_chemical_symbols(), return_counts=True))}
        data = self.ensemble._get_species_counts()
        self.assertDictEqual(data, target)


if __name__ == '__main__':
    unittest.main()
