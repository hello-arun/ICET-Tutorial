import pytest
from mchammer.calculators import ConstituentStrainCalculator
from icet import ClusterSpace, ClusterExpansion
from icet.tools import ConstituentStrain
from ase import Atoms
from ase.build import bulk
import numpy as np
from typing import List
import copy


@pytest.fixture
def primitive_structure() -> Atoms:
    return bulk('Ag', a=4.0)


@pytest.fixture
def supercell(primitive_structure) -> Atoms:
    supercell = primitive_structure.repeat((2, 2, 1))
    supercell[0].symbol = 'Cu'
    return supercell


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
    return lambda k: 2 * k


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


@pytest.fixture
def cluster_expansion(primitive_structure, chemical_symbols):
    cluster_space = ClusterSpace(primitive_structure, [5.0, 3.0], chemical_symbols)
    parameters = [1.2, 0.5, 0.5, -0.2, 0.0, 0.1]
    return ClusterExpansion(parameters=parameters, cluster_space=cluster_space)


@pytest.fixture
def constituent_strain_calculator(constituent_strain, cluster_expansion):
    return ConstituentStrainCalculator(constituent_strain=constituent_strain,
                                       cluster_expansion=cluster_expansion)


def test_init(constituent_strain, cluster_expansion):
    """
    Test that initialization works.
    """
    calc = ConstituentStrainCalculator(constituent_strain=constituent_strain,
                                       cluster_expansion=cluster_expansion)
    assert isinstance(calc.constituent_strain, ConstituentStrain)


@pytest.mark.parametrize("occupations, expected_energy",
                         [(np.array([29, 29, 29, 29]), 3.6),
                          (np.array([47, 29, 29, 29]), 5.0801431977),
                          (np.array([29, 29, 47, 47]), 6.0771033330)])
def test_calculate_total(constituent_strain_calculator, occupations, expected_energy):
    """
    Tests calculation of total energy (strain + CE energy).
    """
    energy = constituent_strain_calculator.calculate_total(occupations=occupations)
    assert abs(energy - expected_energy) < 1e-6


@pytest.mark.parametrize("current_occupations, sites",
                         [(np.array([29, 29, 29, 29]), [0]),
                          (np.array([47, 29, 29, 29]), [0]),
                          (np.array([29, 29, 47, 47]), [0]),
                          (np.array([47, 29, 47, 47]), [2]),
                          (np.array([47, 29, 47, 29]), [1])])
def test_calculate_change_single_flip(constituent_strain_calculator,
                                      current_occupations,
                                      sites):
    """
    Tests calculation of energy change upon a flip (strain + CE energy).
    """
    # First calculate the exepected energy *after* flip
    new_occupations = copy.copy(current_occupations)
    new_site_occupations = []
    for site in sites:
        new_occupation = 29 if current_occupations[site] == 47 else 47
        new_site_occupations.append(new_occupation)
        new_occupations[site] = new_occupation
    e_after = constituent_strain_calculator.calculate_total(occupations=new_occupations)

    # Then energy before
    e_before = constituent_strain_calculator.calculate_total(occupations=current_occupations)

    # Now let's see if we get the same change with custom function
    e_change = constituent_strain_calculator.calculate_change(
        sites=sites,
        current_occupations=current_occupations,
        new_site_occupations=new_site_occupations)
    assert abs(e_change - (e_after - e_before)) < 1e-6

    # We should now be able to do the same thing again
    # since we have not accepted the change
    e_change = constituent_strain_calculator.calculate_change(
        sites=sites,
        current_occupations=current_occupations,
        new_site_occupations=new_site_occupations)
    assert abs(e_change - (e_after - e_before)) < 1e-6

    # Now accept the change and go back
    constituent_strain_calculator.accept_change()
    new_site_occupations_back = [29 if occ == 47 else 47 for occ in new_site_occupations]
    e_change = constituent_strain_calculator.calculate_change(
        sites=sites,
        current_occupations=new_occupations,
        new_site_occupations=new_site_occupations_back)
    assert abs(e_change - (e_before - e_after)) < 1e-6

    # If we try to go back without accepting the change
    # the result should be 'wrong' (this should normally never be done)
    e_change = constituent_strain_calculator.calculate_change(
        sites=sites,
        current_occupations=current_occupations,
        new_site_occupations=new_site_occupations)
    assert abs(e_change - (e_after - e_before)) > 1e-6


@pytest.mark.parametrize("current_occupations, sites",
                         [(np.array([29, 29, 29, 29]), [0, 2])])
def test_calculate_change_multi_flip(constituent_strain_calculator,
                                     current_occupations,
                                     sites):
    """
    Tests behavior when trying to change identity of more than one atom.
    """
    # See what the new occupations would be
    new_site_occupations = []
    for site in sites:
        new_occupation = 29 if current_occupations[site] == 47 else 47
        new_site_occupations.append(new_occupation)

    with pytest.raises(NotImplementedError) as e:
        constituent_strain_calculator.calculate_change(
            sites=sites,
            current_occupations=current_occupations,
            new_site_occupations=new_site_occupations)
    assert 'Only single flips are currently allowed' in str(e.value)
