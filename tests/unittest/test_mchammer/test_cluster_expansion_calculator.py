import pytest

from ase import Atom, Atoms
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators.cluster_expansion_calculator import \
    ClusterExpansionCalculator
import numpy as np
from typing import List


def get_energy_changes(calc: ClusterExpansionCalculator,
                       structure: Atoms,
                       inverted_structure: Atoms,
                       sites: List[int]):
    """
    Calculates change in property upon some change in a structure
    with three different methods.

    Parameters
    ----------
    structure
        Structure to evaluate
    inverted_structure
        A structure whose occupations differ from structure, used to be
        able to know what chemical symbols to switch to
    sites
        Sites in structures on which chemical symbols should change

    Returns
    -------
    float
        Change calculated with local cluster vectors
    float
        Change calculated with global calculation
    float
        Change calculated with custom cluster vector change calculation
    """
    change_local = calc.calculate_change(
        sites=sites,
        current_occupations=structure.get_atomic_numbers(),
        new_site_occupations=inverted_structure.get_atomic_numbers()[sites])

    occupations_before = structure.get_atomic_numbers()
    occupations_after = structure.get_atomic_numbers()
    occupations_after[sites] = inverted_structure.get_atomic_numbers()[sites]
    change_global = calc.calculate_total(occupations=occupations_after) \
        - calc.calculate_total(occupations=occupations_before)

    e_before_ce = calc.cluster_expansion.predict(structure)
    structure_copy = structure.copy()
    structure_copy.set_atomic_numbers(occupations_after)
    e_after_ce = calc.cluster_expansion.predict(structure_copy)
    change_ce = e_after_ce - e_before_ce
    change_ce *= len(structure)

    return change_local, change_global, change_ce


def get_inverted_structure(structure: Atoms, cluster_space: ClusterSpace):
    """
    Make a copy of structure, the occupations of which are changed,
    while still adhering to the constraints of cluster_space

    Parameters
    ----------
    structure
        Structure to be inverted
    cluster_space
        Implcitly defines available alternative symbols on each site

    Returns
    -------
    Atoms
        A structure that differs from structure on all sublattices with
        more than one chemical symbol
    """
    inverted_structure = structure.copy()
    sublattices = cluster_space.get_sublattices(structure)
    for sublattice in sublattices:
        chemical_symbols = sublattice.chemical_symbols

        # Just make sure all symbols are unique
        assert len(set(chemical_symbols)) == len(chemical_symbols)

        if len(chemical_symbols) == 1:
            continue  # Inactive sublattice, nothing to do

        # Now change the identity of each atom in the sublattice
        for ind in sublattice.indices:

            current_symbol = structure[ind].symbol
            assert current_symbol in chemical_symbols

            # Determine a new symbol by using another one from the available chemical symbols
            sym_ind = (chemical_symbols.index(current_symbol) + 1) % len(chemical_symbols)
            new_symbol = chemical_symbols[sym_ind]
            inverted_structure[ind].symbol = new_symbol

    # Make sure we did at least some change
    assert tuple(inverted_structure.get_chemical_symbols()) \
        != tuple(structure.get_chemical_symbols())
    return inverted_structure


def test_get_inverted_structure():
    """
    Tests the above function to make sure it actually changes the structure
    as it is supposed to.
    """
    prim = bulk('Au')
    cs = ClusterSpace(prim, [0.0], ['Au', 'Pd'])
    structure = prim.repeat((2, 1, 1))
    structure[1].symbol = 'Pd'
    inverted_structure = get_inverted_structure(structure, cs)
    assert len(inverted_structure) == len(structure)
    assert len(structure) == 2
    assert structure[0].symbol == 'Au'
    assert inverted_structure[0].symbol == 'Pd'
    assert structure[1].symbol == 'Pd'
    assert inverted_structure[1].symbol == 'Au'


@pytest.fixture
def system(request):
    """
    Constructs a cluster expansion, a supercell, and an "inverted" version
    of that supercell.

    Parameters (in request)
    =======================
    model : str
        string identifier of the system
    repeat : tuple
        Shape of the supercell, passed to ASE:s repeat function
    supercell : str
        such as "pseudorandom", specifies how to occupy the supercell

    Returns
    -------
    ClusterExpansion
        A cluster expansion
    structure
        A supercell of the primitive structure decorated in some fashion
    inverted_structure
        A supercell with the same shape as the other, but in which the occupation
        on each site has been changed, while still adhering to the constraints
        imposed by the cluster space. This structure can be used to facilitate
        flipping of atoms in structure.
    """
    model, repeat, supercell = request.param

    # Create primitive structure and cluster space
    if model == 'binary_fcc':
        alat = 4.0
        chemical_symbols = [['Al', 'Ge']]
        prim = bulk(chemical_symbols[0][0], crystalstructure='fcc', a=alat)
        cutoffs = [7, 6, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
    elif model == 'binary_fcc_merged':
        alat = 4.0
        chemical_symbols = [['Al', 'Ge']]
        prim = bulk(chemical_symbols[0][0], crystalstructure='fcc', a=alat)
        cutoffs = [5, 5]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
        merge_orbits_data = {2: [3], 7: [8, 9]}
        cs.merge_orbits(merge_orbits_data)
    elif model == 'ternary_fcc':
        alat = 4.0
        chemical_symbols = [['Al', 'Ge', 'Ga']]
        prim = bulk(chemical_symbols[0][0], crystalstructure='fcc', a=alat)
        cutoffs = [5, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
    elif model == 'binary_bcc':
        alat = 4.0
        chemical_symbols = [['Al', 'Ge']]
        prim = bulk(chemical_symbols[0][0], crystalstructure='bcc', a=alat)
        cutoffs = [10, 10, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
    elif model == 'binary_hcp':
        alat, clat = 3.4, 5.1
        chemical_symbols = [['Ag', 'Pd'], ['Ag', 'Pd']]
        prim = bulk(chemical_symbols[0][0], a=alat, c=clat, crystalstructure='hcp')
        cutoffs = [5, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
    elif model == 'ternary_hcp':
        alat, clat = 3.4, 5.1
        chemical_symbols = [['Ag', 'Pd', 'Cu'], ['Ag', 'Pd', 'Cu']]
        prim = bulk(chemical_symbols[0][0], a=alat, c=clat, crystalstructure='hcp')
        cutoffs = [5, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
    elif model == 'ternary_hcp_merged':
        alat, clat = 3.4, 5.1
        chemical_symbols = [['Ag', 'Pd', 'Cu'], ['Ag', 'Pd', 'Cu']]
        prim = bulk(chemical_symbols[0][0], a=alat, c=clat, crystalstructure='hcp')
        cutoffs = [5, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
        merge_orbits_data = {1: [3], 4: [5, 6]}
        cs.merge_orbits(merge_orbits_data, ignore_permutations=True)
    elif model == 'sublattices_fcc':
        alat = 4.0
        chemical_symbols = [['Ag', 'Pd'], ['H', 'X']]
        prim = bulk(chemical_symbols[0][0], a=alat, crystalstructure='fcc')
        prim.append(Atom('H', (alat / 2, alat / 2, alat / 2)))
        cutoffs = [5, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
    elif model == 'ternarysublattices_fcc':
        alat = 4.0
        chemical_symbols = [['Ag', 'Pd', 'Cu'], ['H', 'X']]
        prim = bulk(chemical_symbols[0][0], a=alat, crystalstructure='fcc')
        prim.append(Atom('H', (alat / 2, alat / 2, alat / 2)))
        cutoffs = [5, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
    elif model == 'inactivesublattice_fcc':
        alat = 4.0
        chemical_symbols = [['Ag', 'Pd'], ['W']]
        prim = bulk(chemical_symbols[0][0], a=alat, crystalstructure='fcc')
        prim.append(Atom('W', (alat / 2, alat / 2, alat / 2)))
        cutoffs = [5, 4]
        cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=chemical_symbols)
    else:
        raise Exception(f'Unknown model ({model})')

    # Make a supercell as well as an "inverted" version of the supercell,
    # the latter with the purpose of more easily decide how atoms can be changed
    structure = prim.repeat(repeat)
    inverted_structure = get_inverted_structure(structure, cs)

    # Occupy supercell according to some pattern
    if supercell == 'homogeneous':
        pass
    elif supercell == 'pseudorandom':
        for i in [2, 3, 4, 7, 11, 14, 15, 16, 17]:
            if i >= len(structure):
                break
            structure[i].symbol = inverted_structure[i].symbol
        if 'ternary' in model:
            structure[0].symbol = chemical_symbols[0][1]
    elif supercell == 'ordered':
        for i in range(len(structure)):
            if i % 2 == 1:
                continue
            if 'ternary' in model and i % 4 == 0:
                structure[i].symbol = chemical_symbols[0][1]
            else:
                structure[i].symbol = inverted_structure[i].symbol
    elif supercell == 'segregated':
        for i in range(len(structure) // 2):
            if 'ternary' in model:
                if i % 2 == 0:
                    structure[i].symbol = chemical_symbols[0][1]
                else:
                    structure[i].symbol = inverted_structure[i].symbol
            else:
                structure[i].symbol = inverted_structure[i].symbol
    else:
        raise Exception(f'Unknown supercell ({supercell})')

    # Now that we have fiddled with the structure, we need a new inverted_structure
    inverted_structure = get_inverted_structure(structure, cs)

    # Define ECIs that are not all the same
    params = [(-1)**i * ((i + 1) / 10)**1.02 for i in range(len(cs))]

    # Set one to zero so we test pruning
    params[len(params) // 2] = 0

    ce = ClusterExpansion(cluster_space=cs, parameters=params)
    return ce, structure, inverted_structure


# Make a list of parameters; possible combinations of systems and supercells
systems = []
systems_with_calculator_choice = []
for model in ['binary_fcc', 'ternary_fcc', 'binary_bcc', 'ternary_hcp',
              'sublattices_fcc', 'ternarysublattices_fcc', 'inactivesublattice_fcc']:
    for repeat in [(1, 1, 1), (2, 1, 1), (2, 2, 3)]:
        for supercell in ['homogeneous', 'pseudorandom', 'ordered', 'segregated']:
            if repeat in [(1, 1, 1), (2, 1, 1)] and supercell != 'ordered':
                continue
            elif 'ternary' in model and supercell == 'segregated':
                continue
            systems.append(((model, repeat, supercell)))
            systems_with_calculator_choice.append(((model, repeat, supercell), True))
            if model == 'binary_bcc':
                systems_with_calculator_choice.append(((model, repeat, supercell), False))
systems.append((('binary_fcc_merged', (2, 2, 2), 'pseudorandom')))
systems_with_calculator_choice.append((('binary_fcc_merged', (2, 2, 2), 'pseudorandom'), True))
systems.append((('ternary_hcp_merged', (1, 2, 2), 'pseudorandom')))
systems_with_calculator_choice.append((('ternary_hcp_merged', (1, 2, 2), 'pseudorandom'), True))


@pytest.mark.parametrize('system', systems[:5], indirect=['system'])
def test_initialization(system):
    ce, structure, _ = system
    calc = ClusterExpansionCalculator(structure, ce, name='Test CE calc')
    assert isinstance(calc, ClusterExpansionCalculator)
    assert isinstance(calc.cluster_expansion, ClusterExpansion)
    assert calc.name == 'Test CE calc'
    assert abs(calc._property_scaling - len(structure)) < 1e-6
    assert calc.use_local_energy_calculator

    # Some alternative input parameters
    calc = ClusterExpansionCalculator(structure, ce, scaling=5.0, use_local_energy_calculator=False)
    assert isinstance(calc, ClusterExpansionCalculator)
    assert isinstance(calc.cluster_expansion, ClusterExpansion)
    assert calc.name == 'Cluster Expansion Calculator'
    assert abs(calc._property_scaling - 5.0) < 1e-6
    assert not calc.use_local_energy_calculator


@pytest.mark.parametrize('system', systems, indirect=['system'])
def test_get_cluster_vector_against_cluster_space(system):
    """Tests retrieval of full cluster vector from C++ side calculator against
    full cluster vector calculation from cluster space."""
    ce, structure, inverted_structure = system
    calc = ClusterExpansionCalculator(structure, ce, name='Test CE calc')
    cv_calc = calc.cpp_calc.get_cluster_vector(structure.get_atomic_numbers())
    cv_cs = ce.get_cluster_space_copy().get_cluster_vector(structure)
    assert np.allclose(cv_calc, cv_cs)

    # Make sure it works after modifying the structure
    for i in range(min(2, len(structure))):
        structure[i].symbol = inverted_structure[i].symbol
    cv_calc = calc.cpp_calc.get_cluster_vector(structure.get_atomic_numbers())
    cv_cs = ce.get_cluster_space_copy().get_cluster_vector(structure)
    assert np.allclose(cv_calc, cv_cs)


@pytest.mark.parametrize('system, use_local_energy_calculator',
                         systems_with_calculator_choice[:30],
                         indirect=['system'])
def test_change_calculation_flip(system, use_local_energy_calculator):
    """Tests differences when flipping."""
    ce, structure, inverted_structure = system
    calc = ClusterExpansionCalculator(structure, ce, name='Test CE calc',
                                      use_local_energy_calculator=use_local_energy_calculator)
    energy_changes = []
    for i in range(len(structure)):
        sites = [i]
        change_local, change_global, change_ce = \
            get_energy_changes(calc, structure, inverted_structure, sites)
        assert abs(change_local - change_global) < 1e-6
        assert abs(change_global - change_ce) < 1e-6
        energy_changes.append(change_local)

    # Finally make sure we did not test trivial cases only
    if len(structure) > 2:
        assert max(np.abs(energy_changes)) > 0


@pytest.mark.parametrize('system, use_local_energy_calculator',
                         systems_with_calculator_choice,
                         indirect=['system'])
def test_change_calculation_swap(system, use_local_energy_calculator):
    """Tests differences when swapping."""
    ce, structure, inverted_structure = system
    calc = ClusterExpansionCalculator(structure, ce, name='Test CE calc',
                                      use_local_energy_calculator=use_local_energy_calculator)
    energy_changes = []
    for i in range(len(structure)):
        for j in range(3):
            if j >= len(structure) or i == j:
                continue
            sites = [i, j]
            change_local, change_global, change_ce = \
                get_energy_changes(calc, structure, inverted_structure, sites)
            assert abs(change_local - change_global) < 1e-6
            assert abs(change_global - change_ce) < 1e-6
            energy_changes.append(change_local)

    # Finally make sure we did not test trivial cases only
    if len(structure) > 2:
        assert max(np.abs(energy_changes)) > 0


@pytest.mark.parametrize('system, expected_cluster_vector', [
    (('binary_fcc', (2, 2, 3), 'ordered'), [1., 0., -0.22222222, 0.33333333, -0.11111111,
                                            0.33333333, 0., 0., 0., 0., 0., 0., 0., 0.,
                                            0., 0., 0., 0., 0., 0.333333333, -0.22222222, 1.]),
    (('ternary_fcc', (2, 2, 3), 'segregated'), [1., -0.25, 0., 0.0625, 0., -0.0625, -0.5,
                                                0., 0., 0.0625, 0., -0.0208333333, -0.015625,
                                                0., 0.015625, 0, 0.125, 0., 0., 0.0625,
                                                0.0, 0.0]),
    (('binary_hcp', (2, 2, 3), 'ordered'), [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
    (('ternary_hcp', (2, 2, 3), 'ordered'), [1.0, 0.125, 0.2165063509461095, -0.125,
                                             -0.21650635094610976, -0.375, 0.0625,
                                             0.3247595264191646, 0.4375, -0.125,
                                             -0.21650635094610976, -0.3749999999999999,
                                             -0.0625, -0.10825317547305488,
                                             0., 0., -0.0625, -0.10825317547305477,
                                             0.125, 0.10825317547305496, 0.125,
                                             0.3247595264191642, 0.125,
                                             0.10825317547305496, 0.125,
                                             0.3247595264191642]),
    (('sublattices_fcc', (2, 2, 3), 'pseudorandom'), [[1.0, 1/3, 1/6, 1 / 18, 1/9, -1/6,
                                                       1/8, -1/3, 1/3, 1/36, 1/18, -1/9,
                                                       -1/9, -1/18, 1/9, 0.0694444445,
                                                       0.01388888889, 0., -1/6, -1/9, 1/18,
                                                       1/12, 1/12]]),
], indirect=['system'])
def test_get_cluster_vector_against_reference(system, expected_cluster_vector):
    ce, structure, _ = system
    calc = ClusterExpansionCalculator(structure, ce)
    cv = calc.cpp_calc.get_cluster_vector(structure.get_atomic_numbers())
    assert np.allclose(cv, expected_cluster_vector)


@pytest.mark.parametrize('system, expected_local_cluster_vector', [
    (('binary_fcc', (2, 2, 3), 'ordered'), [0.08333333333, -0.08333333, -0.05555555556,
                                            0.166666667, -0.0277777778, -0.04166666667,
                                            0., 0.0833333333, 0.0833333333,
                                            -0.0277777778, -0.09722222222, 0.,
                                            0.05555555556, -0.041666666667,
                                            -0.041666666667, 0.02777777778, 0.0,
                                            0.0138888888889, -0.020833333333,
                                            0.3333333333, -0.111111111, 0.3333333333]),
    (('ternary_fcc', (2, 2, 3), 'segregated'), [0.0833333333, 0.0416666667,
                                                -0.0721687837, -0.02083333333,
                                                0.024056261216, -0.0208333333,
                                                -0.083333333, 0.07216878365, 0.0,
                                                -0.020833333333, 0.0180421959, 0.0,
                                                0.0078125, -0.00751758163,
                                                0.00260416666667, 0.0135316469, 0.0,
                                                -0.0240562612, 0.0240562612,
                                                0.0208333333, 0.0, 0.0]),
    (('binary_hcp', (2, 2, 3), 'ordered'), [0.0416666667, 0.041666667, -0.08333333,
                                            0.083333333, -0.041666666667,
                                            0.125, 0.125]),
    (('ternary_hcp', (2, 2, 3), 'ordered'), [0.04166666667, 0.02083333333, 0.0360843918,
                                             -0.01041666667, -0.018042196, -0.03125,
                                             0.02083333333, 0.03608439, 0.0625,
                                             -0.0104166667, -0.018042196, -0.03125,
                                             -0.0078125, -0.013531647, -0.004510549,
                                             -0.0078125, -0.013020833, -0.022552745,
                                             0.015625, 0.0270632939, 0.046875,
                                             0.081189882, 0.015625, 0.02706329,
                                             0.046875, 0.08118988160]),
    (('sublattices_fcc', (2, 2, 3), 'pseudorandom'), [0.0416666667, 0.0, 0.08333333,
                                                      0.027777777778, 0.0, 0.0, 0.0625, 0.0,
                                                      0.1666666667, 0.0138888889, 0.0,
                                                      -0.01388888889, -0.02777777778,
                                                      -0.0277777778, 0.0555555556,
                                                      0.03472222222, 0.0208333333, 0.0,
                                                      -0.0833333333, 0.0, 0.0277777778,
                                                      0.041666666667, 0.125]),
], indirect=['system'])
def test_get_local_cluster_vector_against_reference(system, expected_local_cluster_vector):
    ce, structure, _ = system
    calc = ClusterExpansionCalculator(structure, ce)
    local_cv = calc.cpp_calc.get_local_cluster_vector(structure.get_atomic_numbers(), 1)
    assert np.allclose(local_cv, expected_local_cluster_vector)


@pytest.mark.parametrize('system, expected_cluster_vector_change', [
    (('binary_fcc', (2, 2, 3), 'ordered'), [0.0, 0.1666666666667, 0.1111111111,
                                            -0.33333333333, 0.05555555556, 0.166666666667,
                                            0.0, -0.1666666666667, -0.1666666666667,
                                            0.05555555555, 0.22222222222, 0.0,
                                            -0.11111111111, 0.11111111111, 0.0,
                                            -0.0555555555556, 0.0, 0.0, 0.1666666666667,
                                            -0.66666666667, 0.222222222, -0.6666666667]),
    (('ternary_fcc', (2, 2, 3), 'segregated'), [0.0, 0.0, 0.144337567, 0.0, -0.0360843918,
                                                0.0416666667, 0.0, -0.144337567, 0.0, 0.0,
                                                -0.0360843918, 0.0, 0.0, 0.00902109796,
                                                -0.0104166667, -0.0270632939, 0.0,
                                                0.0360843918, -0.0721687836, -0.0416666667,
                                                0.0, 0.0]),
    (('binary_hcp', (2, 2, 3), 'ordered'), [0.0, -0.083333333, 0.1666666667,
                                            -0.166666666667, 0.0833333333,
                                            -0.25, -0.25]),
    (('ternary_hcp', (2, 2, 3), 'ordered'), [0.0, -0.0625, -0.0360843918, 0.03125,
                                             0.03608439182, 0.03125, -0.0625, -0.072168784,
                                             -0.0625, 0.03125, 0.0360843918, 0.03125,
                                             0.0234375, 0.0315738428, 0.004510549, 0.0234375,
                                             0.0078125, 0.022552745, -0.046875,
                                             -0.06314768569, -0.078125, -0.081189882,
                                             -0.046875, -0.0631476857, -0.078125,
                                             -0.08118988160]),
    (('sublattices_fcc', (2, 2, 3), 'pseudorandom'), [0.0, 0.0, -0.166666667,
                                                      -0.0555555556, 0.0, 0.0, -0.125, 0.0,
                                                      -0.333333333, -0.0277777778, 0.0,
                                                      0.027777777778, 0.0555555556,
                                                      0.055555555556, -0.111111111,
                                                      -0.069444444, -0.041666666667, 0.0,
                                                      0.166666666667, 0.0, -0.05555555555,
                                                      -0.08333333333, -0.25]),
], indirect=['system'])
def test_get_cluster_vector_change_against_reference(system, expected_cluster_vector_change):
    ce, structure, inverted_structure = system
    calc = ClusterExpansionCalculator(structure, ce)
    cv_change = calc.cpp_calc.get_cluster_vector_change(structure.get_atomic_numbers(), 1,
                                                        inverted_structure.get_atomic_numbers()[1])
    assert np.allclose(cv_change, expected_cluster_vector_change)
