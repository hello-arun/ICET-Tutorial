from custom_functions import (custom_k_to_parameter_function,
                              custom_strain_energy_function)
from icet import ClusterExpansion
from icet.tools import ConstituentStrain
from mchammer.ensembles import VCSGCEnsemble
from mchammer.calculators import ConstituentStrainCalculator
from ase.build import make_supercell
import numpy as np
import os

# Read cluster expansion and define supercell
ce = ClusterExpansion.read('mixing_energy_wo_strain.ce')
prim = ce.get_cluster_space_copy().primitive_structure
P = np.array([[-1, 1, 1],
              [1, -1, 1],
              [1, 1, -1]])
structure = make_supercell(prim, P)
structure = structure.repeat((20, 3, 3))

output_directory = 'monte_carlo_data'
try:
    os.mkdir(output_directory)
except FileExistsError:
    pass

T = 300
for phi in np.arange(0.1, -2.1, -0.1):
    # The constituent strain object can be used standalone
    # but we will use it together with a calucalator
    strain = ConstituentStrain(supercell=structure,
                               primitive_structure=prim,
                               chemical_symbols=['Ag', 'Cu'],
                               concentration_symbol='Cu',
                               strain_energy_function=custom_strain_energy_function,
                               k_to_parameter_function=custom_k_to_parameter_function,
                               damping=10.0)

    # Here we define a calculator to be used with mchammer
    calc = ConstituentStrainCalculator(constituent_strain=strain,
                                       cluster_expansion=ce)

    # Now we can run Monte Carlo in regular fashion
    ens = VCSGCEnsemble(calculator=calc,
                        structure=structure,
                        temperature=T,
                        dc_filename=f'{output_directory}/vcsgc-T{T}-phi{phi:+.3f}.dc',
                        phis={'Cu': phi},
                        kappa=200)
    ens.run(number_of_trial_steps=len(structure) * 10)
    structure = ens.structure  # Use last structure as starting point at next phi
    print(phi, structure)
