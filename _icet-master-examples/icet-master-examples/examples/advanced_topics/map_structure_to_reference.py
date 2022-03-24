"""
This example demonstrates how to map a structure in which the cell has
been scaled and/or the atoms displaced onto an ideal (primitive) structure
"""

# Import modules
from icet.tools import map_structure_to_reference
from ase.build import bulk
from ase import Atom
# End import

# Begin by creating a reference structure, in this case fcc Au.
# Then create a supercell structure, scale the cell and displace the atoms to
# simulate a relaxed structure.
reference = bulk('Au', a=4.00)

supercell = reference.repeat(3)
supercell.rattle(0.1)
supercell.set_cell(1.05 * supercell.cell, scale_atoms=True)

# Switch some atoms to Pd
for i in [0, 1, 5, 8, 10]:
    supercell[i].symbol = 'Pd'

# Map the "relaxed" structure onto an ideal supercell
ideal_structure, info = map_structure_to_reference(supercell, reference)
print('Maximum displacement: {:.3f} Angstrom'.format(info['drmax']))
print('Average displacement: {:.3f} Angstrom'.format(info['dravg']))

# Map a structure that contains vacancies, in this case Pd-Au-H-Vac, in which
# Pd and Au share one sublattice and Pd and H another.
reference = bulk('Au', a=4.00)
reference.append(Atom(('H'), (2.0, 2.0, 2.0)))

supercell = reference.repeat(3)
supercell.rattle(0.1)
supercell.set_cell(1.05 * supercell.cell, scale_atoms=True)

# Switch some Au to Pd and delete some H (to create vacancies)
for i in [0, 4, 6, 2, 7, 3, 17]:
    if supercell[i].symbol == 'Au':
        supercell[i].symbol = 'Pd'
    elif supercell[i].symbol == 'H':
        del supercell[i]

ideal_structure, info = map_structure_to_reference(supercell, reference,
                                                   inert_species=['Au', 'Pd'])
print('Maximum displacement: {:.3f} Angstrom'.format(info['drmax']))
print('Average displacement: {:.3f} Angstrom'.format(info['dravg']))
