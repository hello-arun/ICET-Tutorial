#!/usr/bin/env python3

"""
This file contains unit tests and other tests. It can be executed by
simply executing this file from a shell prompt:

    $ ./test_labeling_generation.py

In which case it will use the system's default python version. If a specific
python version should be used, run that python version with this file as input,
e.g.:

    python3 test_labeling_generation.py

For a description of the python unit testing framework, see this link:
https://docs.python.org/3/library/unittest.html

When executing this file doc testing is also performed on all doc tests in
the structure_container.py file

"""

import unittest

from icet.tools.structure_enumeration_support.labeling_generation \
    import LabelingGenerator, SiteGroup


class TestLabelingGenerator(unittest.TestCase):
    """Container for tests of the LabelingGenerator class."""

    def __init__(self, *args, **kwargs):
        super(TestLabelingGenerator, self).__init__(*args, **kwargs)
        self.iter_species = [(0, 1), (2,), (2,), (2,), (0, 1), (0, 2)]
        self.concentrations = {0: (0, 0.15), 2: (0, 0.3)}

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.lg = LabelingGenerator(self.iter_species,
                                    self.concentrations)

    def test_init(self):
        """Tests that initialization of tested class works."""
        lg = LabelingGenerator(self.iter_species,
                               self.concentrations)
        self.assertIsInstance(lg, LabelingGenerator)

        # Check that site groups were initialized properly
        self.assertEqual(len(lg.site_groups), 3)
        self.assertEqual(lg.site_groups[(0, 1)].multiplicity, 2)
        self.assertEqual(lg.site_groups[(0, 1)].iter_species, (0, 1))

        # Should be able to init also with concentrations=None
        lg = LabelingGenerator(self.iter_species, None)
        self.assertIsInstance(lg, LabelingGenerator)

    def test_yield_products(self):
        """Tests that yield_products works."""

        for site_group in self.lg.site_groups.values():
            site_group.compute_all_combinations(2)

        target_products = [((0, 1, 1, 1), (2, 2, 2, 2, 2, 2), (2, 2)),
                           ((1, 1, 1, 1), (2, 2, 2, 2, 2, 2), (0, 2)),
                           ((1, 1, 1, 1), (2, 2, 2, 2, 2, 2), (2, 2))]
        products = []
        for product in self.lg.yield_products(2):
            products.append(product)

        for target_product, product in zip(target_products, sorted(products)):
            self.assertEqual(product, target_product)

    def test_yield_permutations(self):
        """Tests that yield_permutations works."""
        target_permutations = [((1, 1, 1, 0), (2, 2, 2, 2, 2, 2), (2, 0)),
                               ((1, 1, 1, 0), (2, 2, 2, 2, 2, 2), (0, 2)),
                               ((1, 1, 0, 1), (2, 2, 2, 2, 2, 2), (2, 0)),
                               ((1, 1, 0, 1), (2, 2, 2, 2, 2, 2), (0, 2)),
                               ((1, 0, 1, 1), (2, 2, 2, 2, 2, 2), (2, 0)),
                               ((1, 0, 1, 1), (2, 2, 2, 2, 2, 2), (0, 2)),
                               ((0, 1, 1, 1), (2, 2, 2, 2, 2, 2), (2, 0)),
                               ((0, 1, 1, 1), (2, 2, 2, 2, 2, 2), (0, 2))]

        product = ((0, 1, 1, 1), (2, 2, 2, 2, 2, 2), (0, 2))
        permutations = []
        for labeling in self.lg.yield_permutations(product, 0):
            self.assertIsInstance(labeling, list)
            permutations.append(tuple(labeling))

        for target_per, per in zip(sorted(target_permutations),
                                   sorted(permutations)):
            self.assertEqual(per, target_per)

    def test_sort_labeling(self):
        """Tests that sort_labeling works."""
        labeling = [(0, 1, 1, 1), (2, 2, 2, 2, 2, 2), (0, 2)]
        target_sorted_labeling = (0, 2, 2, 2, 1, 0, 1, 2, 2, 2, 1, 2)
        sorted_labeling = self.lg.sort_labeling(labeling, 2)
        self.assertEqual(target_sorted_labeling, sorted_labeling)

    def test_yield_labeling(self):
        """Tests that yield_labeling works."""
        target_labeling = (1, 2, 2, 2, 1, 2)
        for labeling in self.lg.yield_labelings(1):
            # there should be only one of these
            self.assertEqual(labeling, target_labeling)

        # The generation works a bit differently when concentrations are not
        # specified
        target_labelings = [(0, 2, 2, 2, 0, 0),
                            (0, 2, 2, 2, 0, 2),
                            (0, 2, 2, 2, 1, 0),
                            (0, 2, 2, 2, 1, 2),
                            (1, 2, 2, 2, 0, 0),
                            (1, 2, 2, 2, 0, 2),
                            (1, 2, 2, 2, 1, 0),
                            (1, 2, 2, 2, 1, 2)]
        self.lg.concentrations = None
        labelings = []
        for labeling in self.lg.yield_labelings(1):
            labelings.append(labeling)

        for target_labeling, labeling in zip(sorted(target_labelings),
                                             sorted(labelings)):
            self.assertEqual(labeling, target_labeling)


class TestSiteGroup(unittest.TestCase):
    """Container for tests of the SiteGroup class."""

    def __init__(self, *args, **kwargs):
        super(TestSiteGroup, self).__init__(*args, **kwargs)
        self.iter_species = (0, 1, 3)
        self.position = 3
        self.multiplicity = 7

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.sg = SiteGroup(self.iter_species,
                            self.position)

    def test_init(self):
        """Tests that initialization of tested class works."""
        sg = SiteGroup(self.iter_species,
                       self.position)
        self.assertIsInstance(sg, SiteGroup)
        self.assertEqual(sg.iter_species, self.iter_species)
        self.assertEqual(sg.position, self.position)

    def test_compute_all_combinations(self):
        """Tests that compute_all_combinations works."""
        target_combinations = [(0, 0, 0), (0, 0, 1), (0, 0, 3),
                               (0, 1, 1), (0, 1, 3), (0, 3, 3),
                               (1, 1, 1), (1, 1, 3), (1, 3, 3),
                               (3, 3, 3)]
        self.sg.compute_all_combinations(3)
        for target_comb, comb in zip(sorted(target_combinations),
                                     sorted(self.sg.combinations)):
            self.assertEqual(target_comb, comb)


if __name__ == '__main__':
    unittest.main()
