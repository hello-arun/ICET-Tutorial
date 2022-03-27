from ase import Atoms
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import WangLandauEnsemble
from mchammer.observers import BinaryShortRangeOrderObserver

# Prepare cluster expansion
prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
cs = ClusterSpace(prim, cutoffs=[1.01], chemical_symbols=['Ag', 'Au'])
ce = ClusterExpansion(cs, [0, 0, 2])

# Prepare initial configuration
structure = prim.repeat((4, 4, 1))
for k in range(8):
    structure[k].symbol = 'Ag'

# Set up WL simulation
calculator = ClusterExpansionCalculator(structure, ce)
mc = WangLandauEnsemble(structure=structure,
                        calculator=calculator,
                        energy_spacing=1,
                        dc_filename='wl_n16.dc',
                        ensemble_data_write_interval=len(structure)*10,
                        trajectory_write_interval=len(structure)*100,
                        data_container_write_period=120)

# Add short-range order observer
obs = BinaryShortRangeOrderObserver(cluster_space=cs,
                                    structure=structure,
                                    interval=len(structure)*10,
                                    radius=1.01)
mc.attach_observer(obs)

# Run WL simulation
mc.run(number_of_trial_steps=len(structure)*int(2e5))
