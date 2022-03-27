from ase import Atom
from ase.build import bulk
from icet import ClusterSpace, ClusterExpansion
from icet.tools.structure_generation import occupy_structure_randomly
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import HybridEnsemble
from os import mkdir

# step 1: Set up cluster expansion, structure and calculator
prim = bulk('Pd', a=4.0)
prim.append(Atom('H', position=(2, 2, 2)))
cs = ClusterSpace(prim, cutoffs=[3], chemical_symbols=[('Au', 'Pd'), ('H', 'X')])
ce = ClusterExpansion(cluster_space=cs, parameters=[-0.15, 0, 0, 0, 0.1, 0.05])
structure = prim.repeat(5)
calculator = ClusterExpansionCalculator(structure, ce)

# step 2: Carry out Monte Carlo simulations
# Make sure output directory exists
output_directory = 'monte_carlo_data'
try:
    mkdir(output_directory)
except FileExistsError:
    pass

muH = -0.1
temp = 300
cAu = 0.2
cH_start = 0.2
ensemble_specs = [{'ensemble': 'canonical', 'sublattice_index': 0},
                  {'ensemble': 'semi-grand', 'sublattice_index': 1,
                   'chemical_potentials': {'H': muH, 'X': 0}}]
occupy_structure_randomly(structure=structure,
                          cluster_space=cs,
                          target_concentrations={'A': {'Pd': 1 - cAu, 'Au': cAu},
                                                 'B': {'H': cH_start, 'X': 1 - cH_start}})

# step 3: construct ensemble and run
mc = HybridEnsemble(
    structure=structure,
    calculator=calculator,
    temperature=300,
    ensemble_specs=ensemble_specs,
    dc_filename=f'{output_directory}/hybrid-T{temp}-muH{muH:+.3f}-cAu{cAu:.3f}.dc')

mc.run(number_of_trial_steps=len(structure) * 10)
