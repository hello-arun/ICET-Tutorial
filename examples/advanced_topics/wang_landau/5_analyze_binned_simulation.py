import matplotlib.pyplot as plt
import numpy as np
from mchammer import WangLandauDataContainer
from mchammer.data_containers import (get_average_observables_wl,
                                      get_density_of_states_wl)

# Get density and entropy
dcs = {}
for i in range(6):
    dc = WangLandauDataContainer.read('wl_k{}.dc'.format(i))
    dcs[i] = dc

df, _ = get_density_of_states_wl(dcs)

# Plot density
_, ax = plt.subplots(1, 3)
for i in range(len(ax)):
    ax[i].semilogy(df.energy, df.density / df.density.min())
    ax[i].set_xlabel('Energy')
ax[0].set_ylabel('Density')
ax[0].set_xlim(-140, 110)
ax[0].set_ylim(1e0, 1e18)
ax[1].set_xlim(-130, -90)
ax[1].set_ylim(1e0, 1e7)
ax[2].set_xlim(78, 98)
ax[2].set_ylim(1e0, 1e7)
plt.tight_layout()

plt.savefig('wang_landau_binned_density.svg', bbox_inches='tight')

# Compute thermodynamic averages
df = get_average_observables_wl(dcs,
                                temperatures=np.arange(0.4, 6, 0.05),
                                boltzmann_constant=1)

# Plot reference heat capacity
n_atoms = dcs[i].ensemble_parameters['n_atoms']
_, ax = plt.subplots()
ax.plot(df.temperature, df.potential_std ** 2 / df.temperature ** 2 / n_atoms, lw=5.0,
        color='lightgray', label='reference')

# Determine the heat capacity for different cutoffs
cutoffs = [-80, -40, 0, 40]
for cutoff in cutoffs:
    dcs_cutoff = {}
    for key, dc in dcs.items():
        elr = dc.ensemble_parameters['energy_limit_right']
        if not np.isnan(elr) and elr <= cutoff:
            dcs_cutoff[key] = dc
        else:
            break
    if len(dcs_cutoff) < 1:
        continue
    df = get_average_observables_wl(dcs_cutoff,
                                    temperatures=np.arange(0.4, 6, 0.05),
                                    boltzmann_constant=1)

    # Plot the heat capacity for the given cutoff
    ax.plot(df.temperature, df.potential_std ** 2 / df.temperature ** 2 / n_atoms,
            label=r'$E_c={:0.0f}$'.format(cutoff))

ax.set_xlabel('Temperature')
ax.set_ylabel('Heat capacity')
ax.legend()
plt.savefig('wang_landau_binned_heat_capacity_ecut.svg', bbox_inches='tight')
