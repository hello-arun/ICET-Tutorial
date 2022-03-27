"""
This example demonstrates how to enumerate structures, i.e. how to
generate all inequivalent structures derived from a primitive
structure up to a certain size.
"""

# Import modules
from ase import Atom
from ase.build import bulk, fcc111, add_adsorbate
from ase.db import connect
from icet.tools import enumerate_structures

# Generate all binary fcc structures with up to 6 atoms/cell
# and save them in a database
primitive_structure = bulk('Au')
db = connect('AuPd-fcc.db')
for structure in enumerate_structures(primitive_structure,
                                      range(1, 7),
                                      ['Pd', 'Au']):
    db.write(structure)

# Generate fcc structures in the dilute limit
conc_rest = {'Au': (0, 0.1)}
for structure in enumerate_structures(primitive_structure,
                                      range(10, 14),
                                      ['Pd', 'Au'],
                                      concentration_restrictions=conc_rest):
    db.write(structure)

# Enumerate all palladium hydride structures with up to 4 primitive
# cells (= up to 4 Pd atoms and between 0 and 4 H atoms). We want to
# specify that one site should always be Pd while the other can be
# either a hydrogen or a vacancy ('X' will serve as our vacancy)
a = 4.0
primitive_structure = bulk('Au', a=a)
primitive_structure.append(Atom('H', (a / 2, a / 2, a / 2)))
species = [['Pd'], ['H', 'X']]
db = connect('PdHVac-fcc.db')
for structure in enumerate_structures(primitive_structure, range(1, 5), species):
    db.write(structure)

# Enumerate a copper surface with oxygen adsorbates (or vacancies) in
# fcc and hcp hollow sites.
primitive_structure = fcc111('Cu', (1, 1, 5), vacuum=10.0)
primitive_structure.pbc = [True, True, False]
add_adsorbate(primitive_structure, 'O', 1.2, 'fcc')
add_adsorbate(primitive_structure, 'O', 1.2, 'hcp')
species = []
for atom in primitive_structure:
    if atom.symbol == 'Cu':
        species.append(['Cu'])
    else:
        species.append(['O', 'H'])
db = connect('Cu-O-adsorbates.db')
for structure in enumerate_structures(primitive_structure, range(1, 5), species):
    db.write(structure)
