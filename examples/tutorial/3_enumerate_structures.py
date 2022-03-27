# This scripts runs in about 14 minutes on an i7-6700K CPU.

import matplotlib.pyplot as plt
from numpy import array
from icet import ClusterExpansion
from icet.tools import ConvexHull, enumerate_structures

# step 1: Predict energies for enumerated structures
ce = ClusterExpansion.read('mixing_energy.ce')
species = ['Ag', 'Pd']
data = {'concentration': [], 'mixing_energy': []}
structures = []
cluster_space = ce.get_cluster_space_copy()
chemical_symbols = cluster_space.chemical_symbols
primitive_structure = cluster_space.primitive_structure
for structure in enumerate_structures(structure=primitive_structure,
                                      sizes=range(1, 13),
                                      chemical_symbols=chemical_symbols):
    conc = structure.get_chemical_symbols().count('Pd') / len(structure)
    data['concentration'].append(conc)
    data['mixing_energy'].append(ce.predict(structure))
    structures.append(structure)
print('Predicted energies for {} structures'.format(len(structures)))

# step 2: Construct convex hull
hull = ConvexHull(data['concentration'], data['mixing_energy'])

# step 3: Plot the results
fig, ax = plt.subplots(figsize=(4, 3))
ax.set_xlabel(r'Pd concentration')
ax.set_ylabel(r'Mixing energy (meV/atom)')
ax.set_xlim([0, 1])
ax.set_ylim([-69, 15])
ax.scatter(data['concentration'], 1e3 * array(data['mixing_energy']),
           marker='x')
ax.plot(hull.concentrations, 1e3 * hull.energies, '-o', color='green')
plt.savefig('mixing_energy_predicted.png', bbox_inches='tight')

# step 4: Extract candidate ground state structures
tol = 0.0005
low_energy_structures = hull.extract_low_energy_structures(
    data['concentration'], data['mixing_energy'], tol)
print('Found {} structures within {} meV/atom of the convex hull'.
      format(len(low_energy_structures), 1e3 * tol))
