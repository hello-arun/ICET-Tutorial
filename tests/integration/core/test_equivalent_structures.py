"""
This script checks the cluster vectors calculated for equivalent structures.
"""

import inspect
import os
import numpy as np
from ase.build import bulk
from ase.db import connect
from icet import ClusterSpace


def test_equivalent_structures():

    prim = bulk('Au', a=4.0, crystalstructure='hcp')
    cutoffs = [7.0, 7.0, 7.0]
    chemical_symbols = ['Au', 'Pd']
    cs = ClusterSpace(prim, cutoffs, chemical_symbols)

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    db = connect(os.path.join(
        path, '../../structure_databases/equivalent_structure_pairs.db'))

    # Loop over all pairs
    for structure in db.select():
        # Do not check the pair that was just checked
        if structure.equivalent_structure < structure.id:
            continue

        cv_1 = cs.get_cluster_vector(structure.toatoms())
        cv_2 = cs.get_cluster_vector(
            db.get(structure.equivalent_structure).toatoms())

        assert np.all(np.abs(cv_2 - cv_1) < 1e-6)
