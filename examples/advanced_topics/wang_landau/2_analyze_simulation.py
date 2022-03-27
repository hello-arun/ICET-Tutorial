import matplotlib.pyplot as plt
import numpy as np
from mchammer import WangLandauDataContainer
from mchammer.data_containers import (get_average_observables_wl,
                                      get_density_of_states_wl)

# Read data container
dc = WangLandauDataContainer.read('wl_n16.dc')

# Set up DOS plot
_, ax = plt.subplots()

# Extract and plot the DOS
fill_factor_limits = [10 ** -i for i in range(1, 6)]
for ffl in fill_factor_limits:
    df, _ = get_density_of_states_wl(dc, fill_factor_limit=ffl)

    # Plot DOS
    ax.semilogy(df.energy, df.density, marker='o',
                label=r'$f\leq10^{{{:0.0f}}}$'.format(np.log10(ffl)))

# Add labels and legends
ax.set_xlabel('Energy')
ax.set_ylabel('Density of states')
ax.legend()
plt.savefig('wang_landau_density.svg', bbox_inches='tight')

# Set up plot of heat capacity and short-range order parameter
fig, axes = plt.subplots(nrows=2, sharex=True)
n_atoms = dc.ensemble_parameters['n_atoms']

# Compute thermodynamic averages and plot the results
for f, ffl in enumerate(fill_factor_limits):
    df = get_average_observables_wl(dc,
                                    temperatures=np.arange(0.4, 6, 0.05),
                                    observables=['sro_Ag_1'],
                                    boltzmann_constant=1,
                                    fill_factor_limit=ffl)

    # Plot heat capacity
    line = axes[0].plot(df.temperature,
                        df.potential_std ** 2 / df.temperature ** 2 / n_atoms,
                        label=r'$f\leq10^{{{:0.0f}}}$'.format(np.log10(ffl)))

    # Plot short-range order parameters
    color = line[0].get_color()
    if f == 0:
        axes[1].plot(df.temperature, df.sro_Ag_1_mean, color=color,
                     linestyle='-', label='mean')
        axes[1].plot(df.temperature, df.sro_Ag_1_std, color=color,
                     linestyle='--', label='stddev')
    else:
        axes[1].plot(df.temperature, df.sro_Ag_1_mean, color=color,
                     linestyle='-')
        axes[1].plot(df.temperature, df.sro_Ag_1_std, color=color,
                     linestyle='--')

# Add labels and legends
axes[0].set_xlabel('Temperature')
axes[0].set_ylabel('Heat capacity')
axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Short-range order parameter')
axes[0].legend()
axes[1].legend()
plt.subplots_adjust(hspace=0)
plt.savefig('wang_landau_heat_capacity_sro.svg', bbox_inches='tight')
