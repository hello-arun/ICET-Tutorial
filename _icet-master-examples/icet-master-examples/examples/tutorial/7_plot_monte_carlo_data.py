import matplotlib.pyplot as plt
import pandas as pd

# step 1: Load data frame
dfs = {}
dfs['sgc'] = pd.read_csv('monte-carlo-sgc.csv', delimiter='\t')
dfs['vcsgc'] = pd.read_csv('monte-carlo-vcsgc.csv', delimiter='\t')


# step 1: Plot free energy derivatives
colors = {300: '#D62728',  # red
          900: '#1F77B4'}  # blue
linewidths = {'sgc': 3, 'vcsgc': 1}
alphas = {'sgc': 0.5, 'vcsgc': 1.0}
fig, ax = plt.subplots(figsize=(4, 3.5))
for ensemble, df in dfs.items():
    for T in sorted(df.temperature.unique()):
        df_T = df.loc[df['temperature'] == T].sort_values('Pd_concentration')
        ax.plot(df_T['Pd_concentration'],
                1e3 * df_T['free_energy_derivative'],
                marker='o', markersize=2.5,
                label='{}, {} K'.format(ensemble, T),
                color=colors[T],
                linewidth=linewidths[ensemble], alpha=alphas[ensemble])
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Free energy derivative (meV/atom)')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-600, 500])
ax.legend()
plt.savefig('free_energy_derivative.png', bbox_inches='tight')

# step 2: Plot mixing energy vs composition
df = dfs['sgc']
fig, ax = plt.subplots(figsize=(4, 3.5))
for T in sorted(df.temperature.unique()):
    df_T = df.loc[df['temperature'] == T].sort_values('Pd_concentration')
    e_mix = 1e3 * df_T['mixing_energy']
    e_mix_error = 1e3 * df_T['mixing_energy_error']
    ax.plot(df_T['Pd_concentration'], e_mix,
            marker='o', markersize=2.5, label='{} K'.format(T),
            color=colors[T])
    # Plot error estimate
    ax.fill_between(df_T['Pd_concentration'],
                    e_mix + e_mix_error, e_mix - e_mix_error,
                    color=colors[T], alpha=0.4)
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Mixing energy (meV/atom)')
ax.set_xlim([-0.02, 1.02])
ax.legend()
plt.savefig('mixing_energy_sgc.png', bbox_inches='tight')

# step 3: Plot acceptance ratio vs composition
df = dfs['sgc']
fig, ax = plt.subplots(figsize=(4, 3.5))
for T in sorted(df.temperature.unique()):
    df_T = df.loc[df['temperature'] == T].sort_values('Pd_concentration')
    ax.plot(df_T['Pd_concentration'], df_T['acceptance_ratio'],
            marker='o', markersize=2.5, label='{} K'.format(T),
            color=colors[T])
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Acceptance ratio')
ax.set_xlim([-0.02, 1.02])
ax.legend()
plt.savefig('acceptance_ratio_sgc.png', bbox_inches='tight')
