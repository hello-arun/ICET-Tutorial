import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from icet import ClusterSpace
from icet.tools.structure_generation import generate_target_structure
from mchammer import WangLandauDataContainer
from mchammer.data_containers import get_average_cluster_vectors_wl
from numpy import arange

# Read data container
dc = WangLandauDataContainer.read('wl_n16.dc')

# Set up cluster space
prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
cs = ClusterSpace(prim, cutoffs=[1.01], chemical_symbols=['Ag', 'Au'])

# Get average cluster vectors
df = get_average_cluster_vectors_wl(dc,
                                    cluster_space=cs,
                                    temperatures=arange(0.1, 6.01, 0.1),
                                    boltzmann_constant=1)

# Plot pair
_, ax = plt.subplots()
ax.set_xlabel('Temperature')
ax.set_ylabel('Pair term of cluster vector')
ax.plot(df.temperature, np.array(list(df.cv_std)).T[2], label='stddev')
ax.plot(df.temperature, np.array(list(df.cv_mean)).T[2], label='mean')
ax.legend()
plt.savefig('wang_landau_cluster_vector_vs_temperature.svg', bbox_inches='tight')

# Get low(est) energy structure
temperature = 0.1
target_cluster_vector = list(df[df.temperature == temperature].cv_mean[0])
target_concentrations = {'Ag': 0.5, 'Au': 0.5}
structure = generate_target_structure(cluster_space=cs,
                                      max_size=4,
                                      target_cluster_vector=target_cluster_vector,
                                      target_concentrations=target_concentrations)
