import pytest
from icet.tools import ConstituentStrain, redlich_kister
from icet.tools.constituent_strain import (_generate_k_points,
                                           _ordered_combinations,
                                           _translate_to_1BZ,
                                           _find_equivalent_kpoints)
from icet.tools.constituent_strain_helper_functions import _get_structure_factor
from ase.build import bulk
from ase import Atoms
from typing import List
import numpy as np
import copy


@pytest.fixture
def primitive_structure() -> Atoms:
    return bulk('Ag', a=4.0)


@pytest.fixture
def supercell(primitive_structure) -> Atoms:
    return primitive_structure.repeat((2, 2, 1))


@pytest.fixture
def chemical_symbols() -> List[str]:
    return ['Ag', 'Cu']


@pytest.fixture
def concentration_symbol() -> str:
    return 'Cu'


@pytest.fixture
def strain_energy_function():
    return lambda params, c: abs(params[2]) * c


@pytest.fixture
def k_to_parameter_function():
    return lambda kpt: 2 * kpt


@pytest.fixture
def constituent_strain(supercell,
                       primitive_structure,
                       chemical_symbols,
                       concentration_symbol,
                       strain_energy_function,
                       k_to_parameter_function) -> ConstituentStrain:
    return ConstituentStrain(supercell=supercell,
                             primitive_structure=primitive_structure,
                             chemical_symbols=chemical_symbols,
                             concentration_symbol=concentration_symbol,
                             strain_energy_function=strain_energy_function,
                             k_to_parameter_function=k_to_parameter_function)


def test_spin_variables(constituent_strain):
    """Tests initialization of spin variables and concentration definition."""
    assert len(constituent_strain.spin_variables) == 2
    assert constituent_strain.spin_variables[47] == 1
    assert constituent_strain.spin_variables[29] == -1
    assert constituent_strain.spin_up == 47
    assert constituent_strain.concentration_number == 29


@pytest.mark.parametrize("k_to_parameter_function, expected_parameter",
                         [(lambda kpt: 2 * kpt, 0.5), (lambda kpt: -kpt, 0.25), (None, 0.25)])
def test_kpoints(supercell, primitive_structure, chemical_symbols, concentration_symbol,
                 strain_energy_function, k_to_parameter_function, expected_parameter):
    """Test that kpoints were properly intitialized and work as expected."""
    constituent_strain = ConstituentStrain(supercell=supercell,
                                           primitive_structure=primitive_structure,
                                           chemical_symbols=chemical_symbols,
                                           concentration_symbol=concentration_symbol,
                                           strain_energy_function=strain_energy_function,
                                           k_to_parameter_function=k_to_parameter_function)

    assert len(constituent_strain.kpoints) == 6

    # Make sure that k=[0, 0, 1/4] is present and extract it
    kdiffs = [np.linalg.norm(kpoint.kpt - np.array([0, 0, 0.25]))
              for kpoint in constituent_strain.kpoints]
    assert min(kdiffs) < 1e-9
    kpoint = constituent_strain.kpoints[np.argmin(kdiffs)]

    # Test damping
    expected_damping = np.exp(-(1.0 * np.linalg.norm([0, 0, 0.25])) ** 2)
    assert np.abs(kpoint.damping_factor - expected_damping) < 1e-9

    # Test that the strain energy function returns the expected value
    c = 0.3
    expected_strain = expected_parameter * c
    assert np.abs(kpoint.strain_energy_function(c) - expected_strain) < 1e-9


def test_nonbinary_systems(supercell, primitive_structure, concentration_symbol,
                           strain_energy_function, k_to_parameter_function):
    """Tests that initialization fails if a nonbinary system is specified."""
    kwargs = dict(supercell=supercell,
                  primitive_structure=primitive_structure,
                  concentration_symbol=concentration_symbol,
                  strain_energy_function=strain_energy_function,
                  k_to_parameter_function=k_to_parameter_function)

    with pytest.raises(NotImplementedError) as e:
        ConstituentStrain(chemical_symbols=['Ag', 'Cu', 'Au'], **kwargs)
    assert 'only works for binary' in str(e.value)

    with pytest.raises(ValueError) as e:
        ConstituentStrain(chemical_symbols=['Ag'], **kwargs)
    assert 'specify two' in str(e.value)


@pytest.mark.parametrize("occupations, expected_c",
                         [([29, 29, 47, 47], 0.5),
                          ([47, 29, 47, 47], 0.25),
                          ([47, 47, 47, 47], 0.0)])
def test_get_concentration(constituent_strain, occupations, expected_c):
    """Tests retrieval of concentration."""
    c = constituent_strain.get_concentration(np.array(occupations))
    assert abs(c - expected_c) < 1e-6


@pytest.mark.parametrize("occupations, kpt, expected_value",
                         [(np.array([29] * 4), np.array([0, 0, 0.25]), 0),
                          (np.array([29, 47, 47, 29]), np.array([0, 0, 0.25]), -1),
                          (np.array([47, 29, 29, 47]), np.array([0, 0, 0.25]), 1),
                          (np.array([47, 29, 29, 47]), np.array([-0.125, 0.125, 0.125]), 0),
                          (np.array([29, 29, 47, 47]), np.array([-0.125, 0.125, 0.125]), -1),
                          (np.array([47, 29, 47, 47]), np.array([-0.125, 0.125, 0.125]), -0.5),
                          (np.array([47, 47, 47, 47]), np.array([-0.125, 0.125, 0.125]), 0)])
def test_get_structure_factor(supercell, occupations, kpt, expected_value):
    """Tests structure factor calculation."""
    positions = supercell.get_positions()

    S_pos = _get_structure_factor(occupations=occupations,
                                  positions=positions,
                                  kpt=kpt,
                                  spin_up=47)
    assert abs(S_pos - expected_value) < 1e-6

    # If we flip the spins the structure factor should change sign
    S_neg = _get_structure_factor(occupations=occupations,
                                  positions=positions,
                                  kpt=kpt,
                                  spin_up=29)
    assert abs(S_pos + S_neg) < 1e-6


@pytest.mark.parametrize("wanted_kpoint, occupations, concentration, "
                         "expected_DE, expected_structure_factor, multiplicity",
                         [([0, 0, 0.25], [29] * 4, 1.0, 0.0, 0.0, 0.5),
                          ([0, 0, 0.25], [47] * 4, 0.0, 0.0, 0.0, 0.5),
                          ([0, 0, 0.25], [29, 47, 47, 29], 0.5, 0.25, -1, 0.5),
                          ([-0.125, 0.125, 0.125], [47, 29, 47, 47], 0.25, 0.0625, -0.5, 0.5)])
def test_get_constituent_strain_term(constituent_strain, wanted_kpoint,
                                     occupations, concentration,
                                     expected_DE, expected_structure_factor,
                                     multiplicity):
    """Tests calculation of constituent strain term."""
    # Extract wanted k point
    kdiffs = [np.linalg.norm(kpoint.kpt - np.array(wanted_kpoint))
              for kpoint in constituent_strain.kpoints]
    assert min(kdiffs) < 1e-9
    kpoint = constituent_strain.kpoints[np.argmin(kdiffs)]

    cs_term = constituent_strain._get_constituent_strain_term(kpoint,
                                                              np.array(occupations),
                                                              concentration)

    # We calculate damping separately since the actual numbers are awkward
    expected_value = expected_DE * expected_structure_factor**2 \
        * multiplicity * np.exp(-np.linalg.norm(wanted_kpoint)**2)
    enumerator = (4 * concentration * (1 - concentration))
    if abs(enumerator) > 1e-6:
        expected_value /= enumerator
    assert abs(cs_term - expected_value) < 1e-6
    assert abs(kpoint.structure_factor - expected_structure_factor) < 1e-6


@pytest.mark.parametrize("occupations, expected_value",
                         [([29] * 4, 0),
                          ([47] * 4, 0),
                          ([47, 47, 47, 29], 0.078900822032611),
                          ([47, 29, 47, 29], 0.119275833246148)])
def test_get_constituent_strain(constituent_strain, occupations, expected_value):
    """Tests calculation of full constituent strain."""
    e = constituent_strain.get_constituent_strain(np.array(occupations))
    assert abs(e - expected_value) < 1e-6


@pytest.mark.parametrize("occupations, atom_index, expected_change",
                         [([47] * 4, 3, 0.078900822032611),
                          ([47, 47, 47, 29], 1, 0.119275833246148 - 0.078900822032611),
                          ([47, 29, 47, 29], 1, 0.078900822032611 - 0.119275833246148)])
def test_get_constituent_strain_change(constituent_strain, occupations,
                                       atom_index, expected_change):
    """
    Tests calculation of change in constituent strain energy
    upon flip of one atom. Also test the accept_change function.
    """
    # Since this function relies on proper values in kpoint.structure_factor,
    # we first need to make a total strain calculation
    occupations = np.array(occupations)

    # Calculate change
    constituent_strain.get_constituent_strain(occupations)
    de = constituent_strain.get_constituent_strain_change(occupations, atom_index)
    assert abs(de - expected_change) < 1e-6

    # Now let's *not* accept the change and try again
    # We should then get the same result
    de = constituent_strain.get_constituent_strain_change(occupations, atom_index)
    assert abs(de - expected_change) < 1e-6

    # Let's accept the change. We should then be able to make the opposite
    # flip and get the same result but with changed sign
    constituent_strain.accept_change()
    new_occupation = 47 if occupations[atom_index] == 29 else 29
    new_occupations = copy.copy(occupations)
    new_occupations[atom_index] = new_occupation
    de = constituent_strain.get_constituent_strain_change(new_occupations, atom_index)
    assert abs(de + expected_change) < 1e-6

    # Let's go back again
    constituent_strain.accept_change()
    de = constituent_strain.get_constituent_strain_change(occupations, atom_index)
    assert abs(de - expected_change) < 1e-6

    # Finally make sure we do not get the same result if we do not accept the change;
    # this should normally not be done
    de = constituent_strain.get_constituent_strain_change(new_occupations, atom_index)
    assert abs(de + expected_change) > 1e-6


def test_generate_k_points_fcc(primitive_structure):
    """
    Test generation of k points for small FCC cell.
    """
    supercell = primitive_structure.repeat((2, 1, 1))
    kpoints = [k for k in _generate_k_points(primitive_structure, supercell, tol=1e-6)]
    assert len(kpoints) == 3

    expected_kpoints = [([0, 0, 0], 1),
                        ([0.125, -0.125, -0.125], 0.5),
                        ([-0.125, 0.125, 0.125], 0.5)]
    for expected_kpoint in expected_kpoints:
        assert np.any([np.linalg.norm(expected_kpoint[0] - kpoint[0]) < 1e-6
                       and abs(expected_kpoint[1] - kpoint[1]) < 1e-6
                       for kpoint in kpoints])


def test_generate_k_points_sc():
    """
    Test generation of k points for a slightly larger simple cubic cell.
    """
    primitive_structure = bulk('Ag', a=4.0, crystalstructure='sc')
    supercell = primitive_structure.repeat((2, 2, 1))
    kpoints = [k for k in _generate_k_points(primitive_structure, supercell, tol=1e-6)]
    assert len(kpoints) == 9
    expected_kpoints = [([0, 0, 0], 1),
                        ([0.125, 0, 0], 0.5),  # on a face -> multiplicity 2
                        ([-0.125, 0, 0], 0.5),
                        ([0, 0.125, 0], 0.5),
                        ([0, -0.125, 0], 0.5),
                        ([0.125, 0.125, 0], 0.25),  # corner -> multiplicity 4
                        ([0.125, -0.125, 0], 0.25),
                        ([-0.125, 0.125, 0], 0.25),
                        ([-0.125, -0.125, 0], 0.25)]
    for expected_kpoint in expected_kpoints:
        assert np.any([np.linalg.norm(expected_kpoint[0] - kpoint[0]) < 1e-6
                       and abs(expected_kpoint[1] - kpoint[1]) < 1e-6
                       for kpoint in kpoints])


def test_ordered_combinations():
    """
    Test generation of tuples of integers to be used when
    searching for k points.
    """
    ordered_combinations = [tuple(i) for i in _ordered_combinations(3, 2)]
    expected_combinations = [(0, 3), (3, 0), (1, 2), (2, 1)]
    assert tuple(sorted(ordered_combinations)) == tuple(sorted(expected_combinations))


@pytest.mark.parametrize("k, translations",
                         [([0, 0, 0], [0, 0, 0]),
                          ([0, 0, 0], [-10, 7, 0]),
                          ([0, 0, 0.03], [0, 0, 0]),
                          ([0, 0, 0.03], [-3, -6, -2]),
                          ([0.12, 0.03123, 0.13], [0, 0, 0]),
                          ([0.12, 0.03123, 0.13], [1, 0, 0]),
                          ([0.12, 0.03123, 0.13], [10, 10, -10])])
def test_translate_to_1BZ(primitive_structure, k, translations):
    """Test translation of arbitrary k point back to 1BZ."""
    reciprocal_cell = np.linalg.inv(primitive_structure.cell)

    # Translate out from 1BZ
    k_translated = np.array(k) + np.dot(reciprocal_cell, translations)

    # Try to bring it back
    k_back = _translate_to_1BZ(k_translated, reciprocal_cell, tol=1e-6)
    assert np.linalg.norm(k_back - k) < 1e-6


@pytest.mark.parametrize("kpt, equivalent_kpts",
                         [([0, 0, 0], [[0, 0, 0]]),  # gamma
                          ([0, 0, 0.03], [[0, 0, 0.03]]),  # random
                          ([0, 0, 0.25], [[0, 0, 0.25], [0, 0, -0.25]]),  # 100 face
                          ([0.125, 0.125, 0.125], [[0.125, 0.125, 0.125],  # 111 face
                                                   [-0.125, -0.125, -0.125]]),
                          ([0.250, 0.125, 0], [[0.250, 0.125, 0],  # corner
                                               [0, -0.125, 0.250],
                                               [0, -0.125, -0.250],
                                               [-0.250, 0.125, 0]]),
                          ([0.250, 0.125 / 2, 0.125 / 2],  # edge
                           [[0.250, 0.125 / 2, 0.125 / 2],
                            [-0.250, 0.125 / 2, 0.125 / 2],
                            [0, -0.1875, -0.1875]])])
def test_find_equivalent_kpoints_fcc(primitive_structure, kpt, equivalent_kpts):
    """
    Test identification of equivalent k-points in a FCC cell 1BZ
    (which is a RTO).
    """
    reciprocal_cell = np.linalg.inv(primitive_structure.cell)
    kpoints = _find_equivalent_kpoints(kpt, reciprocal_cell, tol=1e-6)

    assert len(kpoints) == len(equivalent_kpts)
    for kpoint in kpoints:
        assert np.any([np.linalg.norm(np.array(equivalent_kpt) - kpoint) < 1e-6
                       for equivalent_kpt in equivalent_kpts])


@pytest.mark.parametrize("kpt, equivalent_kpts",
                         [([0, 0, 0], [[0, 0, 0]]),  # gamma
                          ([0, 0, 0.03], [[0, 0, 0.03]]),  # random
                          ([0, 0, 0.125], [[0, 0, 0.125], [0, 0, -0.125]]),  # face
                          ([0, 0.125, 0.125], [[0, 0.125, 0.125],  # edge
                                               [0, 0.125, -0.125],
                                               [0, -0.125, 0.125],
                                               [0, -0.125, -0.125]]),
                          ([0.125, 0.125, 0.125], [[-0.125, -0.125, -0.125],  # corner
                                                   [0.125, 0.125, -0.125],
                                                   [0.125, -0.125, 0.125],
                                                   [-0.125, 0.125, 0.125],
                                                   [0.125, -0.125, -0.125],
                                                   [-0.125, 0.125, -0.125],
                                                   [-0.125, -0.125, 0.125],
                                                   [0.125, 0.125, 0.125]])])
def test_find_equivalent_kpoints_sc(kpt, equivalent_kpts):
    """
    Test identification of equivalent k-points in a simple cubic cell 1BZ.
    """
    primitive_structure = bulk('Ag', a=4.0, crystalstructure='sc')
    reciprocal_cell = np.linalg.inv(primitive_structure.cell)
    kpoints = _find_equivalent_kpoints(kpt, reciprocal_cell, tol=1e-6)

    assert len(kpoints) == len(equivalent_kpts)
    for kpoint in kpoints:
        assert np.any([np.linalg.norm(np.array(equivalent_kpt) - kpoint) < 1e-6
                       for equivalent_kpt in equivalent_kpts])


@pytest.mark.parametrize("x, parameters, expected_value",
                         [(0.0, [0.1, 0.2, 0.3], 0),
                          (1.0, [0.1, 0.2, 0.3], 0),
                          (0.5, [1, 2, 3], 0.25),
                          (0.25, [1, 2], 0.375),
                          (0.25, [0], 0),
                          (1 / 3, [1, -2, 3, 0.5, 1], 0.1550068587)])
def test_redlich_kister(x, parameters, expected_value):
    """
    Test evaluation of Redlich-Kister polynomial.
    """
    val = redlich_kister(x, *np.array(parameters))
    assert np.abs(val - expected_value) < 1e-6
