import pytest
from mchammer.observers import ConstituentStrainObserver
from icet.tools import ConstituentStrain
from ase import Atoms
from ase.build import bulk
from typing import List


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
def constituent_strain_observer(constituent_strain):
    return ConstituentStrainObserver(constituent_strain=constituent_strain)


def test_init(constituent_strain):
    """Tests that initialization works."""
    obs = ConstituentStrainObserver(constituent_strain=constituent_strain,
                                    interval=333)
    assert isinstance(obs.constituent_strain, ConstituentStrain)
    assert obs.interval == 333
    assert obs.tag == 'ConstituentStrainObserver'


def test_get_observable(constituent_strain_observer, supercell):
    """Tests that observable is returned properly."""
    obs = constituent_strain_observer.get_observable(structure=supercell)
    assert isinstance(obs, dict)
    assert 'constituent_strain_energy' in obs
    assert abs(obs['constituent_strain_energy'] - 0.315603288) < 1e-6

    # Change decoration
    supercell[1].symbol = 'Cu'
    obs = constituent_strain_observer.get_observable(structure=supercell)
    assert abs(obs['constituent_strain_energy'] - 0.477103333) < 1e-6

    # Change back
    supercell[1].symbol = 'Ag'
    obs = constituent_strain_observer.get_observable(structure=supercell)
    assert abs(obs['constituent_strain_energy'] - 0.315603288) < 1e-6
