import itertools

import numpy as np
from ase import Atom
from ase.build import bulk
from trainstation import Optimizer

from icet.tools import Constraints, get_mixing_energy_constraints
from icet import ClusterSpace


def test_constraints():

    np.random.seed(42)
    n, m = 20, 10

    A = np.random.random((n, m))
    y = np.random.random(n)

    # constraint sum eci[inds] = 0
    inds1 = [1, 3, 4, 5]
    inds2 = [2, 6, 7, 8]
    M = np.zeros((2, m))
    M[0, inds1] = 1
    M[1, inds2] = 1

    c = Constraints(m)
    c.add_constraint(M)

    Ac = c.transform(A)
    opt = Optimizer((Ac, y), fit_method='ridge')
    opt.train()

    parameters = c.inverse_transform(opt.parameters)
    sum_1 = parameters[inds1].sum()
    sum_2 = parameters[inds2].sum()
    print('constraints 1, ', sum_1)
    print('constraints 2, ', sum_2)

    assert abs(sum_1) < 1e-12
    assert abs(sum_2) < 1e-12

    # Test get_mixing_energy_constraints function
    a = 4.0
    prim = bulk('Au', a=a)
    prim.append(Atom('H', position=(a / 2, a / 2, a / 2)))
    cs = ClusterSpace(prim, cutoffs=[7.0, 5.0, 4.0],
                      chemical_symbols=[['Au', 'Pd', 'Cu'], ['H', 'V']])

    A = np.random.random((n, len(cs)))
    y = np.random.random(n)  # Add 10 so we are sure nothing gets zero by accident

    c = get_mixing_energy_constraints(cs)
    Ac = c.transform(A)
    opt = Optimizer((Ac, y), fit_method='ridge')
    opt.train()

    parameters = c.inverse_transform(opt.parameters)

    for syms in itertools.product(['Au', 'Pd', 'Cu'], ['H', 'V']):
        prim.set_chemical_symbols(syms)
        assert abs(np.dot(parameters, cs.get_cluster_vector(prim))) < 1e-12

    # Test get_mixing_energy_constraints for structure with multiple atoms in sublattice
    prim = bulk('Ti', crystalstructure='hcp')
    chemical_symbols = ['Ti', 'W', 'V']
    cs = ClusterSpace(prim, cutoffs=[7.0, 5.0, 4.0],
                      chemical_symbols=chemical_symbols)

    A = np.random.random((n, len(cs)))
    y = np.random.random(n) + 10.0  # Add 10 so we are sure nothing gets zero by accident

    c = get_mixing_energy_constraints(cs)
    Ac = c.transform(A)
    opt = Optimizer((Ac, y), fit_method='ridge')
    opt.train()

    parameters = c.inverse_transform(opt.parameters)

    for sym in chemical_symbols:
        prim.set_chemical_symbols([sym, sym])
        assert abs(np.dot(parameters, cs.get_cluster_vector(prim))) < 1e-12

    prim.set_chemical_symbols(['Ti', 'W'])
    assert abs(np.dot(parameters, cs.get_cluster_vector(prim))) > 1e-12
