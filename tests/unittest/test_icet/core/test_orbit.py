import unittest

from icet.core.lattice_site import LatticeSite
from icet.core.structure import Structure
from _icet import Cluster
from icet.core.orbit import Orbit
from ase.build import bulk
import numpy as np


class TestOrbit(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestOrbit, self).__init__(*args, **kwargs)

        self.structure = Structure.from_atoms(bulk('Al'))

        self.allowed_permutations_pair = set([(0, 1)])
        self.allowed_permutations_triplet = set([(0, 2, 1)])

        lattice_site_for_cluster = [LatticeSite(0, [i, 0, 0]) for i in range(3)]

        # Pair and triplet clusters
        self.pair_sites = [lattice_site_for_cluster[0],
                           lattice_site_for_cluster[1]]
        self.pair_sites_alt = [lattice_site_for_cluster[1],
                               lattice_site_for_cluster[2]]
        self.triplet_sites = lattice_site_for_cluster
        self.pair_cluster = Cluster(self.pair_sites, self.structure)
        self.pair_cluster_alt = Cluster(self.pair_sites, self.structure)
        self.triplet_cluster = Cluster(self.triplet_sites, self.structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.structure.allowed_atomic_numbers = [[13, 30]]

        self.orbit_pair = Orbit([self.pair_cluster],
                                self.allowed_permutations_pair)

        self.orbit_pair_two_clusters = Orbit([self.pair_cluster, self.pair_cluster_alt],
                                             self.allowed_permutations_pair)

        self.orbit_triplet = Orbit([self.triplet_cluster],
                                   self.allowed_permutations_triplet)

    def test_init(self):
        """Tests the initializer."""
        orbit = Orbit([self.pair_cluster], self.allowed_permutations_pair)
        self.assertIsInstance(orbit, Orbit)

        orbit = Orbit([self.pair_cluster, self.pair_cluster_alt],
                      self.allowed_permutations_pair)
        self.assertIsInstance(orbit, Orbit)

        orbit = Orbit([self.triplet_cluster], self.allowed_permutations_triplet)
        self.assertIsInstance(orbit, Orbit)

    def test_clusters(self):
        """Tests getting the clusters of an orbit."""
        self.assertEqual(len(self.orbit_pair.clusters), 1)
        cluster = self.orbit_pair.clusters[0]
        self.assertEqual(len(cluster.lattice_sites), len(self.pair_sites))
        for ls_ret, ls_tgt in zip(cluster.lattice_sites, self.pair_sites):
            self.assertEqual(ls_ret, ls_tgt)

    def test_representative_cluster(self):
        """Tests getting the representative cluster of orbit."""
        representative_cluster = self.orbit_pair.representative_cluster
        self.assertEqual(representative_cluster.order, 2)
        self.assertEqual(representative_cluster.order, self.orbit_pair.order)
        self.assertAlmostEqual(representative_cluster.radius, 1.4318912319)
        self.assertAlmostEqual(representative_cluster.radius, self.orbit_pair.radius)
        for site_ret, site_tgt in zip(representative_cluster.lattice_sites, self.pair_sites):
            self.assertEqual(site_ret, site_tgt)

        representative_cluster = self.orbit_triplet.representative_cluster
        self.assertEqual(representative_cluster.order, 3)
        self.assertEqual(representative_cluster.order, self.orbit_triplet.order)
        self.assertAlmostEqual(representative_cluster.radius, 1.909188309)
        self.assertAlmostEqual(representative_cluster.radius, self.orbit_triplet.radius)
        for site_ret, site_tgt in zip(representative_cluster.lattice_sites, self.triplet_sites):
            self.assertEqual(site_ret, site_tgt)

    def test_order(self):
        """Tests getting the order of orbit."""
        self.assertEqual(
            self.orbit_pair.order, 2)
        self.assertEqual(
            self.orbit_pair_two_clusters.order, 2)
        self.assertEqual(
            self.orbit_triplet.order, 3)

    def test_len(self):
        """Tests lenght of orbit."""
        self.assertEqual(len(self.orbit_pair), 1)
        self.assertEqual(len(self.orbit_pair_two_clusters), 2)
        self.assertEqual(len(self.orbit_triplet), 1)

    def test_radius(self):
        """Tests geometrical size of orbit."""
        self.orbit_pair.radius

    def test_translate(self):
        """Tests translation by an offset."""
        offset = np.array((1, 1, 1))

        for clusters, perm in zip([(self.pair_cluster,),
                                   (self.pair_cluster, self.pair_cluster_alt),
                                   (self.triplet_cluster,)],
                                  [self.allowed_permutations_pair,
                                   self.allowed_permutations_pair,
                                   self.allowed_permutations_triplet]):
            orbit = Orbit(clusters, perm)
            orbit_with_offset = Orbit(clusters, perm)
            orbit_with_offset.translate(offset)

            # Loop through sites check index is same and offset has been added
            for cluster, cluster_with_offset in zip(orbit.clusters, orbit_with_offset.clusters):
                for site, site_with_offset in zip(cluster.lattice_sites,
                                                  cluster_with_offset.lattice_sites):
                    self.assertEqual(site.index, site_with_offset.index)
                    self.assertListEqual(list(site.unitcell_offset),
                                         list(site_with_offset.unitcell_offset - offset))

            # List is allowed but only length 3
            with self.assertRaises(TypeError):
                orbit_with_offset.translate([1, 1, 1, 1])

    def test_allowed_permutations(self):
        """Tests the allowed permutations property."""
        self.assertEqual(self.orbit_pair.allowed_permutations,
                         [list(i) for i in self.allowed_permutations_pair])
        self.assertEqual(self.orbit_triplet.allowed_permutations,
                         [list(i) for i in self.allowed_permutations_triplet])

    def test_get_multicomponent_vectors_pairs(self):
        """Tests the get mc vectors functionality for a pair orbit."""
        # Binary mc vectors
        # Allow only identity permutation
        self.structure.allowed_atomic_numbers = [[13, 14]]
        allowed_permutations = set([tuple(i for i in range(self.orbit_pair.order))])
        orbit = Orbit([Cluster(self.pair_sites, self.structure)], allowed_permutations)
        mc_vectors = [el['multicomponent_vector'] for el in orbit.cluster_vector_elements]
        self.assertEqual(mc_vectors, [[0, 0]])

        # Ternary mc vectors
        self.structure.allowed_atomic_numbers = [[13, 14, 15]]
        allowed_permutations = set([tuple(i for i in range(self.orbit_pair.order))])
        orbit = Orbit([Cluster(self.pair_sites, self.structure)], allowed_permutations)
        mc_vectors = [el['multicomponent_vector'] for el in orbit.cluster_vector_elements]
        target = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.assertEqual(mc_vectors, target)

        # Allow the [1,0] permutation
        allowed_permutations = set([(0, 1), (1, 0)])
        orbit = Orbit([Cluster(self.pair_sites, self.structure)], allowed_permutations)
        mc_vectors = [el['multicomponent_vector'] for el in orbit.cluster_vector_elements]
        target = [[0, 0], [0, 1], [1, 1]]
        self.assertEqual(mc_vectors, target)

    def test_get_multicomponent_vectors_triplets(self):
        """Tests  the get mc vectors functionality for a triplet orbit."""
        # Binary mc vectors
        # Allow only identity permutation
        allowed_permutations = set([tuple(i for i in range(self.orbit_triplet.order))])
        self.structure.allowed_atomic_numbers = [[13, 14, 15]]
        orbit = Orbit([Cluster(self.triplet_sites, self.structure)], allowed_permutations)
        mc_vectors = [el['multicomponent_vector'] for el in orbit.cluster_vector_elements]
        target = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        self.assertEqual(mc_vectors, target)

        # Allow the [0, 2, 1] permutation
        allowed_permutations = set([(0, 1, 2), (0, 2, 1)])
        orbit = Orbit([Cluster(self.triplet_sites, self.structure)], allowed_permutations)
        mc_vectors = [el['multicomponent_vector'] for el in orbit.cluster_vector_elements]
        target = [[0, 0, 0], [0, 0, 1], [0, 1, 1],
                  [1, 0, 0], [1, 0, 1], [1, 1, 1]]
        self.assertEqual(mc_vectors, target)


if __name__ == '__main__':
    unittest.main()
