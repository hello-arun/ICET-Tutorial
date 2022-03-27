from ase.build import make_supercell
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import VCSGCEnsemble
import numpy as np
from os import mkdir

# step 1: Set up structure to simulate as well as calculator
ce = ClusterExpansion.read('mixing_energy.ce')
structure = make_supercell(ce.get_cluster_space_copy().primitive_structure,
                           3 * np.array([[-1, 1, 1],
                                         [1, -1, 1],
                                         [1, 1, -1]]))
calculator = ClusterExpansionCalculator(structure, ce)

# step 2: Carry out Monte Carlo simulations
# Make sure output directory exists
output_directory = 'monte_carlo_data'
try:
    mkdir(output_directory)
except FileExistsError:
    pass
for temperature in [900, 300]:
    # Evolve configuration through the entire composition range
    for phi in np.arange(-2.1, 0.11, 0.08):
        # Initialize MC ensemble
        mc = VCSGCEnsemble(
            structure=structure,
            calculator=calculator,
            temperature=temperature,
            dc_filename='{}/vcsgc-T{}-phi{:+.3f}.dc'.format(output_directory, temperature, phi),
            phis={'Pd': phi},
            kappa=200)

        mc.run(number_of_trial_steps=len(structure) * 30)
        structure = mc.structure
