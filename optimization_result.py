import itertools
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import differential_evolution
import InitParams

BREAKPOINT = 90
MAX_LPSR = 0
PENALTY_COEFF = 1e6
PENALTY_COEFF_SURPLUS = 0
OBJECTIVE_FUN = -3 # -3 for ASC, -4 for master thesis function, -2 only deficit and loss

# Path to the input file generated with opt_fun_to_excel
results_for_all_designs = 'doe_2706.csv'

# Load the 'designs_modified' file
results = np.loadtxt(results_for_all_designs)

# Define targets
targets = {
    
    # 'Energy Deficit': results.iloc[:, 4],
    # 'Energy Loss': results.iloc[:, 5],
    # 'Cost': results.iloc[:, 6]
}

# Features
features = results[:, 0:4]
quadratic_combinations = list(itertools.combinations_with_replacement(range(features.shape[1]), 2))
feat_quad = np.array([features[:, i:i+1] * features[:, j:j+1] for i, j in quadratic_combinations])
feat_quad = np.concatenate(feat_quad, axis=1)
lpsr_values = results[:, results.shape[1] - 1]
# print(lpsr_values)

features = np.concatenate((features, feat_quad), axis=1)
# features.columns = feature_labels + quadratic_terms + cubic_terms
# Fit quadratic regression models for each target

models = {}
target_values = results[:,OBJECTIVE_FUN]
target_lspr = results[:,-1]
target_surplus = results[:, -6]

# Add a constant term to the features
features_const = sm.add_constant(features)

# Fit the model
model = sm.OLS(target_values, features_const).fit()
model_lspr = sm.OLS(target_lspr, features_const).fit()
model_surplus = sm.OLS(target_surplus, features_const).fit()

# Display bar chart for coefficients
coefficients = model.params[1:]  # Exclude the intercept
sorted_coefficients_idx = np.argsort(np.abs(coefficients), axis=0)[::-1]
sorted_coefficients = np.abs(coefficients[sorted_coefficients_idx])
feature_labels = ['PV', 'B', 'rSOC', 'tank']
quadratic_terms = [f"{feature_labels[i]}*{feature_labels[j]}" for i, j in quadratic_combinations]
# Bar labels
bar_labels = feature_labels + quadratic_terms
bar_labels = np.array(bar_labels)
sorted_labels = bar_labels[sorted_coefficients_idx]

# # Bar plot
ax1 = plt.gca()
ax1.bar(sorted_labels, sorted_coefficients, label=r'Magnitude of effects $\beta$')
ax1.set_xticklabels(sorted_labels, rotation=45, ha="right")
ax1.set_ylabel(r'Magnitude of effects $\beta_{i,j}$', color='blue')
ax1.tick_params(axis='x', which='both', bottom=False)
ax1.tick_params(axis='y', colors='blue')


# # Calculate cumulative sum and normalize to get percentages
cumsum_percentages = np.cumsum(sorted_coefficients) / sorted_coefficients.sum() * 100

# Line plot for cumulative percentages on a secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Cumulative Percentage [\%]', color='red')
ax2.plot(sorted_labels, cumsum_percentages, color='red', linestyle='--', marker='o', label='Cumulative Percentage')
ax2.tick_params(axis='y', colors='red')
ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.savefig('paretoasc.pdf', format='pdf', bbox_inches='tight')

# # Find the index where cumulative percentage exceeds 80%
# exceed_index = np.argmax(cumsum_percentages > BREAKPOINT)

# # # If all values are below the breakpoint, set the index to the last one
# if exceed_index == 0:
#     exceed_index = len(sorted_labels) - 1

# # Mark the point on the line plot with a vertical line
# ax2.axvline(x=exceed_index, color='green', linestyle='-', label=f"Exceeds {BREAKPOINT}%")

# ax1.set_xlabel('Features')
# ax1.set_ylabel('Magnitude of Coefficients')
# ax2.set_ylabel('Cumulative Percentage')
# plt.title(f'Magnitude of Coefficients and Cumulative Percentage')
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.show()


# Display summary statistics
print(f"Summary Statistics for Quadratic Regression Model:\n\n{model.summary()}\n\n")
    
# Predictions for the current target
predictions = model.predict(features_const)


def f(X):
        quadratic_combinations = list(itertools.combinations_with_replacement(range(len(X)), 2))
        features = np.array(X)[np.newaxis,:]
        feat_quad = np.array([features[:, i:i + 1] * features[:, j:j + 1] for i, j in quadratic_combinations])
        feat_quad = np.concatenate(feat_quad, axis=1)

        features = np.concatenate((features, feat_quad), axis=1)
        features_const = sm.add_constant(features, has_constant="add")

        f = model.predict(features_const)

        lpsr = model_lspr.predict(features_const)
        max_lpsr = MAX_LPSR
        penalty = 0
        if lpsr > max_lpsr:
                penalty = PENALTY_COEFF * (lpsr - max_lpsr)

        surplus = model_surplus.predict(features_const)
        surplus_penalty = PENALTY_COEFF_SURPLUS * surplus
        # surplus_penalty = 0
        return f + penalty + surplus_penalty

bounds = [(-1,1),(-1,1),(-1,1),(-1,1)]
result = differential_evolution(f, bounds)
print(result.x, result.fun)

def denormalize_value(normalized_value, original_min, original_max):
    return original_min + (original_max - original_min) * (normalized_value + 1) / 2

def denormalize(normalized_values, ranges):
    denormalized_values = []
    for i, value in enumerate(normalized_values):
        denormalized_values.append(denormalize_value(value, ranges[i][0], ranges[i][1]))
    return denormalized_values

# Given ranges
ranges = [(16, 48), (3, 23), (6, 30), (8, 61)]

# Denormalize
denormalized_values = denormalize(result.x, ranges)
print(denormalized_values)

# import seaborn as sns


# # Scatter regression plot
# sns.regplot(x=predictions, y=results[:,-1], scatter_kws={'s': 30}, line_kws={'color': 'red'})

# plt.title('Regression plot for the objective function')
# plt.xlabel('Predicted Values')
# plt.ylabel('Observed Values')
# plt.show()

# # Adjust layout to prevent clipping of the legend:
# plt.tight_layout(pad=1.5)  # You can adjust the pad value according to your preference
# # Save the combined plot to a PDF file:
# plt.savefig('regression.pdf', format='pdf', bbox_inches='tight')