from multiprocessing import Pool
from ase.build import make_supercell
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
import numpy as np

# step 1: Set up structure to simulate as well as calculator
ce = ClusterExpansion.read('mixing_energy.ce')
cs = ce.get_cluster_space_copy()
structure = make_supercell(cs.primitive_structure,
                           3 * np.array([[-1, 1, 1],
                                         [1, -1, 1],
                                         [1, 1, -1]]))
calculator = ClusterExpansionCalculator(structure, ce)


# step 2: Define a function that handles MC run of one set of parameters
def run_mc(args):
    temperature = args['temperature']
    dmu = args['dmu']
    mc = SemiGrandCanonicalEnsemble(
        structure=structure,
        calculator=calculator,
        temperature=temperature,
        dc_filename='sgc-T{}-dmu{:+.3f}.dc'.format(temperature, dmu),
        chemical_potentials={'Ag': 0, 'Pd': dmu})
    mc.run(number_of_trial_steps=len(structure) * 30)


# step 3: Define all sets of parameters to be run
args = []
for temperature in range(600, 199, -100):
    for dmu in np.arange(-0.6, 0.6, 0.05):
        args.append({'temperature': temperature, 'dmu': dmu})

# step 4: Define a Pool object with the desired number of processes and run
pool = Pool(processes=4)
results = pool.map_async(run_mc, args)
results.get()
