from typing import Dict
import pytest
from ase import Atoms
from ase.build import bulk


@pytest.fixture(scope='session', autouse=True)
def structures_for_testing() -> Dict[str, Atoms]:
    """
    Dictionary of structures used for testing.
    """

    structures = {}

    structures['Al-fcc-primitive_cell'] = bulk('Al', 'fcc', a=1.0)
    structures['Al-fcc-supercell'] = structure = bulk('Al', 'fcc', a=1.0).repeat(2)

    structure = bulk('Al', 'fcc', a=1.0).repeat(2)
    structure.rattle(stdev=0.001, seed=42)
    structures['Al-fcc-distorted'] = structure

    structure = bulk('Ti', 'bcc', a=1.0).repeat(2)
    structure.symbols[[a.index for a in structure if a.index % 2 == 0]] = 'W'
    structures['WTi-bcc-supercell'] = structure

    structures['NaCl-rocksalt-cubic-cell'] = bulk('NaCl', 'rocksalt', a=1.0)

    structures['Ni-hcp-hexagonal-cell'] = bulk('Ni', 'hcp', a=0.625, c=1.0)

    a = 1.0
    b = 0.5 * a
    structure = Atoms('BaZrO3',
                      positions=[(0, 0, 0), (b, b, b),
                                 (b, b, 0), (b, 0, b), (0, b, b)],
                      cell=[a, a, a], pbc=True)
    structures['BaZrO3-perovskite'] = structure

    return structures
