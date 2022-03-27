import unittest

from ase.build import bulk
from icet import ClusterSpace
from mchammer.observers import BinaryShortRangeOrderObserver


class TestBinaryShortRangeOrderObserver(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestBinaryShortRangeOrderObserver,
              self).__init__(*args, **kwargs)

        self.structure = bulk('Au').repeat([2, 2, 1])
        self.structure[1].symbol = 'Pd'
        self.structure = self.structure.repeat(2)

        cutoffs = [5]
        subelements = [['Au', 'Pd']]*len(self.structure)
        self.cs = ClusterSpace(self.structure, cutoffs, subelements)
        self.interval = 10
        self.radius = 5

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_init_with_ternary(self):
        """Tests the init with ternaries (should fail)."""
        # First a simple ternary
        subelements = [['Au', 'Pd', 'H']]*len(self.structure)
        cutoffs = [3]

        cluster_space = ClusterSpace(self.structure, cutoffs, subelements)

        with self.assertRaises(ValueError) as msg:
            BinaryShortRangeOrderObserver(
                cluster_space=cluster_space, structure=self.structure,
                interval=self.interval, radius=self.radius)

        self.assertIn('Cluster space has more than two allowed species on a',
                      str(msg.exception))

        # Ternary with a binary sublattice
        subelements = [['Au', 'Pd', 'H'], ['Be', 'Ba']]*(len(self.structure)//2)
        cluster_space = ClusterSpace(self.structure, cutoffs, subelements)
        with self.assertRaises(ValueError) as msg:
            BinaryShortRangeOrderObserver(
                cluster_space=cluster_space, structure=self.structure,
                interval=self.interval, radius=self.radius)
        self.assertIn('Cluster space has more than two allowed species on a',
                      str(msg.exception))

        # Double Ternary
        subelements = [['Au', 'Pd', 'H'], [
            'Be', 'Ba', 'Ge']]*(len(self.structure)//2)
        cluster_space = ClusterSpace(self.structure, cutoffs, subelements)
        with self.assertRaises(ValueError) as msg:
            BinaryShortRangeOrderObserver(
                cluster_space=cluster_space, structure=self.structure,
                interval=self.interval, radius=self.radius)
        self.assertIn('Cluster space has more than two allowed species on a',
                      str(msg.exception))

    def test_init_with_sublattices(self):
        """Tests the init of sublattices for different cases."""
        # First a binary with inactice sublattice (should work)
        subelements = [['Au', 'Pd'], ['H']]*(len(self.structure)//2)
        cutoffs = [3]
        cluster_space = ClusterSpace(self.structure, cutoffs, subelements)
        BinaryShortRangeOrderObserver(
            cluster_space=cluster_space, structure=self.structure,
            interval=self.interval, radius=self.radius)

        # First a binary with several inactice sublattice (should work)
        subelements = [['Au'], ['O'], ['Ge', 'Pd'], ['H']]*(len(self.structure)//4)
        cutoffs = [3]
        cluster_space = ClusterSpace(self.structure, cutoffs, subelements)
        BinaryShortRangeOrderObserver(
            cluster_space=cluster_space, structure=self.structure,
            interval=self.interval, radius=self.radius)

        # Two binary sublattices (should fail)
        subelements = [['Au', 'H'], ['Be', 'Ge']]*(len(self.structure)//2)
        cluster_space = ClusterSpace(self.structure, cutoffs, subelements)
        with self.assertRaises(ValueError) as msg:
            BinaryShortRangeOrderObserver(
                cluster_space=cluster_space, structure=self.structure,
                interval=self.interval, radius=self.radius)
        self.assertIn('Number of binary sublattices must equal one, not 2',
                      str(msg.exception))

    def test_sro_values(self):
        """Tests the sro values for a few systems where it is easily
        done by hand.
        """
        structure = bulk('Au').repeat([2, 2, 2])
        observer = BinaryShortRangeOrderObserver(
            cluster_space=self.cs, structure=structure,
            interval=self.interval, radius=self.radius)

        # 1 Pd in pure Au sro
        structure.set_chemical_symbols(['Au'] * len(structure))
        structure[0].symbol = 'Pd'
        sro_parameters = observer.get_observable(structure)
        conc_Au = (len(structure) - 1)/len(structure)
        # In total there will be 12 Pd neighboring an Au atom

        Z_Pd = 12.0
        N_Au = len(structure) - 1
        # Number of bonds per atom
        multiplicity = 6.0
        expected_sro = 1 - Z_Pd/(2*multiplicity*N_Au*(1-conc_Au))
        self.assertEqual(expected_sro, sro_parameters['sro_Au_1'])

        # 1 Au in Pure Pd sro
        structure.set_chemical_symbols(['Pd'] * len(structure))
        structure[0].symbol = 'Au'
        sro_parameters = observer.get_observable(structure)
        conc_Au = 1/len(structure)
        N_Au = 1

        # That one Au will have 12 Pd in its shell
        Z_Pd = 12
        multiplicity = 6.0
        expected_sro = 1 - Z_Pd/(2*multiplicity*N_Au*(1-conc_Au))
        self.assertEqual(expected_sro, sro_parameters['sro_Au_1'])

        structure = bulk('Au', cubic=True).repeat([10, 10, 1])
        structure.wrap()
        observer = BinaryShortRangeOrderObserver(
            cluster_space=self.cs, structure=structure,
            interval=self.interval, radius=self.radius)

        for atom in structure:
            if atom.position[1] > 4*5:
                atom.symbol = 'Pd'
        sro_parameters = observer.get_observable(structure)

    def test_checkerboard_sro(self):
        """Tests the SRO parameter values for a BCC structure with
        checkerboard occupation.
        """

        structure = bulk('Al', 'bcc', cubic=True, a=4)
        structure[1].symbol = 'Ge'
        structure = structure.repeat(6)
        cutoffs = [5]
        subelements = [['Al', 'Ge']]*len(structure)
        cluster_space = ClusterSpace(structure, cutoffs, subelements)

        observer = BinaryShortRangeOrderObserver(
            cluster_space=cluster_space, structure=structure,
            interval=self.interval, radius=self.radius)

        sro_parameters = observer.get_observable(structure)
        self.assertEqual(sro_parameters['sro_Al_1'], -1.0)
        self.assertEqual(sro_parameters['sro_Al_2'], 1.0)

    def setUp(self):
        """Set up observer before each test."""
        self.observer = BinaryShortRangeOrderObserver(
            cluster_space=self.cs, structure=self.structure,
            interval=self.interval, radius=self.radius)

    def test_property_tag(self):
        """Tests property tag."""
        self.assertEqual(self.observer.tag, "BinaryShortRangeOrderObserver")

    def test_property_interval(self):
        """Tests property interval."""
        self.assertEqual(self.observer.interval, self.interval)

    def test_get_concentrations(self):
        conc = self.observer._get_concentrations(self.structure)

        self.assertEqual(conc['Au'], 0.75)
        self.assertEqual(conc['Pd'], 0.25)


if __name__ == '__main__':
    unittest.main()
