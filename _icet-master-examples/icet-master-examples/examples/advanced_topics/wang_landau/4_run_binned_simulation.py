from multiprocessing import Pool
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import WangLandauEnsemble
from mchammer.ensembles.wang_landau_ensemble import get_bins_for_parallel_simulations


# Define a function that runs a WL simulation for one set of parameters
def run_simulation(args: dict) -> None:
    mc = WangLandauEnsemble(structure=args['structure'],
                            calculator=calculator,
                            energy_spacing=args['energy_spacing'],
                            energy_limit_left=args['energy_limit_left'],
                            energy_limit_right=args['energy_limit_right'],
                            fill_factor_limit=args['fill_factor_limit'],
                            flatness_limit=args['flatness_limit'],
                            dc_filename=args['dc_filename'],
                            ensemble_data_write_interval=args['ensemble_data_write_interval'],
                            trajectory_write_interval=args['trajectory_write_interval'],
                            data_container_write_period=args['data_container_write_period'])
    mc.run(number_of_trial_steps=args['number_of_trial_steps'])


# Prepare cluster expansion
prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
cs = ClusterSpace(prim, cutoffs=[1.01], chemical_symbols=['Ag', 'Au'])
ce = ClusterExpansion(cs, [0, 0, 2])

# Prepare initial configuration
structure = prim.repeat((8, 8, 1))
for k in range(len(structure) // 2):
    structure[k].symbol = 'Ag'

# Initalize calculator
calculator = ClusterExpansionCalculator(structure, ce)

# Define parameters for simulations that correspond to different bins
energy_spacing = 1
bins = get_bins_for_parallel_simulations(n_bins=6, energy_spacing=energy_spacing,
                                         minimum_energy=-128, maximum_energy=96)
args = []
for k, (energy_limit_left, energy_limit_right) in enumerate(bins):
    args.append({'structure': structure,
                 'energy_spacing': energy_spacing,
                 'energy_limit_left': energy_limit_left,
                 'energy_limit_right': energy_limit_right,
                 'fill_factor_limit': 1e-4,
                 'flatness_limit': 0.7,
                 'dc_filename': 'wl_k{}.dc'.format(k),
                 'ensemble_data_write_interval': len(structure)*100,
                 'trajectory_write_interval': len(structure)*1000,
                 'data_container_write_period': 120,
                 'number_of_trial_steps': len(structure)*int(2e9)})

# Initiate a Pool object with the desired number of processes and run
pool = Pool(processes=4)
results = pool.map_async(run_simulation, args)
results.get()
