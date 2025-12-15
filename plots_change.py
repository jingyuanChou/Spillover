import matplotlib.pyplot as plt

# Define the data for coverage rate and spillover ratio
coverage_rate = {'0-5': {'+1': [0.43125,0.18838], '+10': [0.53,0.103], '+15': [0.9534,0.1170], '+20': [0.8553,0.1153], '+30': [0.832,0.1145], '+40': [0.56,0.212]},
                 '0-10': {'+1': [0.511, 0.181], '+10': [0.5895,0.1124], '+15': [0.908,0.1049], '+20': [0.9062,0.109], '+30': [0.8646,0.097], '+40': [0.6393,0.227]},
                 '0-15': {'+1': [0.5821,0.19], '+10': [0.8301,0.116], '+15': [0.7575,0.114], '+20': [0.8577,0.097], '+30': [0.8387,0.109], '+40': [0.6648,0.112]},
                 '0-20': {'+1': [0.611, 0.308], '+10': [0.7903,0.1114], '+15': [0.9667,0.1218], '+20': [0.9557,0.1148], '+30': [0.868,0.113], '+40': [0.6468,0.114]}}

spillover_ratio = {'0-5': {'+1': [0,0], '+10': [0.64,0.09], '+15': [1.02,0.1109], '+20': [1.4051,0.2203], '+30': [1.4226,0.2920], '+40': [2.27,0.81]},
                   '0-10': {'+1': [0.06,0.032], '+10': [0.9613,0.197], '+15': [1.0245,0.0938], '+20': [1.0077,0.0753], '+30': [1.2863,0.176], '+40': [2.1918,0.634]},
                   '0-15': {'+1': [0.6065,0.2508], '+10': [0.9301,0.175], '+15': [1.0827,0.2712], '+20': [1.1384,0.1624], '+30': [1.0564,0.1834], '+40': [4.0745,1.7815]},
                   '0-20': {'+1': [0.8291,0.3288], '+10': [0.9198,0.1107], '+15': [0.9217,0.066], '+20': [1.0129,0.0628], '+30': [1.6188,0.3423], '+40': [4.022,1.338]}}

# Define the figure size and subplots
fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

# Set the x-axis label for all subplots

# Set the y-axis label for the first subplot
axs[0].set_ylabel('Ratio')

# Loop through the different intervals and plot the coverage rate and spillover ratio
for i, interval in enumerate(coverage_rate.keys()):
    x_values = list(coverage_rate[interval].keys())
    y_values_coverage_rate = [values[0] for values in coverage_rate[interval].values()]
    y_values_spillover_ratio = [values[0] for values in spillover_ratio[interval].values()]

    # Get the standard deviations
    y_std_coverage_rate = [values[1] for values in coverage_rate[interval].values()]
    y_std_spillover_ratio = [values[1] for values in spillover_ratio[interval].values()]

    # Plot the coverage rate with error bars
    axs[i].errorbar(x_values, y_values_coverage_rate, yerr=y_std_coverage_rate, label='Coverage Rate')

    axs[i].set_xlabel(f'Treatment Interval {interval}')

    # Plot the spillover ratio on the same subplot with error bars
    axs[i].errorbar(x_values, y_values_spillover_ratio, yerr=y_std_spillover_ratio, color='orange', label='Spillover Ratio')

    # Emphasize the range between +15 and +30 on the x-axis with two dashed lines
    axs[i].axvline(x='+15', ymin=0, ymax=1, color='gray', linestyle='--')
    axs[i].axvline(x='+30', ymin=0, ymax=1, color='gray', linestyle='--')
    axs[i].axhline(y=1, color='gray', linestyle='--')

    # Set the legend for the subplot
    axs[i].legend(loc='upper right')

plt.subplots_adjust(wspace=0.1)
plt.show()
