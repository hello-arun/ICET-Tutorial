import numpy as np
from mchammer import DataContainer
from mchammer.data_analysis import get_autocorrelation_function
import matplotlib.pyplot as plt


dc = DataContainer.read('dc_size6_nPd691_T400.dc')

# getting data
steps, E_mix = dc.get('mctrial', 'potential')

# analyze of potential
summary = dc.analyze_data('potential')
print(summary)
corr_length = summary['correlation_length']

# correlation function
max_lag = 150  # window size for correlation
acf = get_autocorrelation_function(E_mix, max_lag=max_lag)


fig = plt.figure(figsize=(4.2, 2.5))
ax1 = fig.add_subplot(111)

ax1.plot(steps[:max_lag], acf, label='potential')
ax1.axhline(y=np.exp(-2), ls='--', c='k', label=r'e$^{-2}$')
ax1.annotate('Correlation length', xy=(corr_length, np.exp(-2)), xytext=(corr_length, 0.5),
             arrowprops=dict(arrowstyle="->"))

plt.xlim([0, steps[max_lag]])
ax1.set_xlabel('mctrial steps')
plt.ylabel('ACF')
plt.legend()

plt.tight_layout()
plt.savefig('autocorrelation.svg')
plt.show(block=False)
