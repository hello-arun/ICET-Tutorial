from custom_functions import (custom_k_to_parameter_function,
                              custom_strain_energy_function)
from icet import (ClusterSpace,
                  StructureContainer,
                  ClusterExpansion,
                  Optimizer)
from icet.tools import (ConstituentStrain,
                        enumerate_structures,
                        get_mixing_energy_constraints)
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.constraints import UnitCellFilter
import numpy as np


def get_relaxed_energy(structure):
    calc = EMT()
    structure = structure.copy()  # Since we will relax it
    structure.set_calculator(calc)
    ucf = UnitCellFilter(structure)
    qn = QuasiNewton(ucf)
    qn.run(fmax=0.05)
    return structure.get_potential_energy()


# Calculate energy of elements so we can define mixing energy properly
a = 4.0
prim = bulk('Ag', a=a)
elemental_energies = {}
for element in ['Ag', 'Cu']:
    structure = prim.copy()
    structure[0].symbol = element
    elemental_energies[element] = get_relaxed_energy(structure)

# Define a cluster space
cutoffs = a * np.array([1.2, 0.8])
cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=[['Ag', 'Cu']])

# Fill a structure container with data
sc = StructureContainer(cluster_space=cs)
for structure in enumerate_structures(prim, range(6), ['Ag', 'Cu']):
    conc = structure.get_chemical_symbols().count('Cu') / len(structure)

    energy = get_relaxed_energy(structure)
    mix_energy = energy / len(structure) \
        - conc * elemental_energies['Cu'] - (1 - conc) * elemental_energies['Ag']

    strain = ConstituentStrain(supercell=structure,
                               primitive_structure=prim,
                               chemical_symbols=['Ag', 'Cu'],
                               concentration_symbol='Cu',
                               strain_energy_function=custom_strain_energy_function,
                               k_to_parameter_function=custom_k_to_parameter_function,
                               damping=10.0)
    strain_energy = strain.get_constituent_strain(structure.get_atomic_numbers())

    sc.add_structure(structure, properties={'concentration': conc,
                                            'mix_energy': mix_energy,
                                            'mix_energy_wo_strain': mix_energy - strain_energy})

# Constrain sensing matrix so as to reproduce c=0 and c=1 exactly,
# then fit a cluster expansion
A, y = sc.get_fit_data(key='mix_energy_wo_strain')
constr = get_mixing_energy_constraints(cs)
A_constrained = constr.transform(A)

opt = Optimizer((A_constrained, y))
opt.train()
parameters = constr.inverse_transform(opt.parameters)
ce = ClusterExpansion(cluster_space=cs, parameters=parameters, metadata=opt.summary)
ce.write('mixing_energy_wo_strain.ce')
