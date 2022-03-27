"""
Tests the behavior of the cluster space class given different combinations
of tolerance parameters.
"""

from ase.build import bulk
from icet import ClusterSpace
from numpy import allclose


def test_structure_tolerances():

    def test_structures(cluster_space, structures):
        results = []
        for k, row in enumerate(structures):
            cv_ref = cluster_space.get_cluster_vector(row['no_noise'])
            for key, structure in row.items():
                if key == 'no_noise':
                    continue
                try:
                    cv = cluster_space.get_cluster_vector(structure)
                    results.append(allclose(cv, cv_ref))
                except Exception as e:
                    if 'Try increasing symprec' in str(e):
                        results.append(False)
                    else:
                        results.append(None)

        return results

    # primitive structure
    prim = bulk('Au', a=1, crystalstructure='bcc', cubic=True)

    # structures with varying degree of noise
    structures = []

    # structure 1
    rec = {}
    structures.append(rec)

    structure = prim.repeat(2)
    structure[1].symbol = 'Ag'
    structure[7].symbol = 'Ag'
    rec['no_noise'] = structure

    structure = rec['no_noise'].copy()
    structure[0].position += [0, 0, 0.00001]
    structure[4].position += [0.00001, 0, 0]
    rec['noise_1e-5'] = structure

    structure = rec['no_noise'].copy()
    structure[1].position += [0, 0, 0.0001]
    structure[3].position += [0, 0.0001, 0]
    rec['noise_1e-4'] = structure

    # structure 2
    rec = {}
    structures.append(rec)

    structure = prim.repeat(2)
    structure[3].symbol = 'Ag'
    structure[5].symbol = 'Ag'
    structure[1].symbol = 'Ag'
    structure[1].symbol = 'Ag'
    structure[7].symbol = 'Ag'
    rec['no_noise'] = structure

    structure = rec['no_noise'].copy()
    structure[5].position += [0, 0, 0.00001]
    structure[7].position += [0.00001, 0, 0]
    rec['noise_1e-5'] = structure

    structure = rec['no_noise'].copy()
    structure[2].position += [0, 0, 0.0001]
    structure[6].position += [0, 0.0001, 0]
    rec['noise_1e-4'] = structure

    # set up one cluster space with lower precision
    cs1 = ClusterSpace(structure=prim,
                       cutoffs=[1.2, 0.9],
                       chemical_symbols=['Au', 'Ag'],
                       symprec=1e-3)

    # set up another cluster space with tighter precision
    cs2 = ClusterSpace(structure=prim,
                       cutoffs=[1.5, 0.9999],
                       chemical_symbols=['Au', 'Ag'],
                       symprec=1e-4)

    # test computing the cluster vectors of the structures
    result = test_structures(cs1, structures)
    target = [True, True, True, True]
    for k, (r, t) in enumerate(zip(result, target)):
        assert r == t, f'failed for case 1: {k}'

    result = test_structures(cs2, structures)
    target = [True, False, True, False]
    for k, (r, t) in enumerate(zip(result, target)):
        assert r == t, f'failed for case 2: {k}'
