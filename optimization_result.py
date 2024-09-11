import itertools
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import differential_evolution
import InitParams


MAX_LPSR = 0
PENALTY_COEFF = 1e6
PENALTY_COEFF_SURPLUS = 0
# OBJECTIVE_FUN = -4 # -3 for ASC, -4 for master thesis function, -2 only deficit and loss

# Path to the input file generated with opt_fun_to_excel
results_for_all_designs = 'doe_test.csv'

# Load the 'designs_modified' file
results = np.loadtxt(results_for_all_designs)

def plot_pareto_chart(model, feature_labels, quadratic_combinations, output_filename='pareto.pdf'):
    # Extract coefficients (excluding the intercept)
    coefficients = model.params[1:]  # Exclude the intercept
    sorted_coefficients_idx = np.argsort(np.abs(coefficients), axis=0)[::-1]
    sorted_coefficients = np.abs(coefficients[sorted_coefficients_idx])

    # Create labels for quadratic terms
    quadratic_terms = [f"{feature_labels[i]}*{feature_labels[j]}" for i, j in quadratic_combinations]

    # Bar labels (linear + quadratic)
    bar_labels = feature_labels + quadratic_terms
    bar_labels = np.array(bar_labels)
    sorted_labels = bar_labels[sorted_coefficients_idx]

    # Bar plot for the magnitude of effects
    ax1 = plt.gca()
    ax1.bar(sorted_labels, sorted_coefficients, label=r'Magnitude of effects $\beta$')
    ax1.set_xticklabels(sorted_labels, rotation=45, ha="right")
    ax1.set_ylabel(r'Magnitude of effects $\beta_{i,j}$', color='blue')
    ax1.tick_params(axis='x', which='both', bottom=False)
    ax1.tick_params(axis='y', colors='blue')

    # Calculate cumulative sum and normalize to get percentages
    cumsum_percentages = np.cumsum(sorted_coefficients) / sorted_coefficients.sum() * 100

    # Line plot for cumulative percentages on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Percentage [\%]', color='red')
    ax2.plot(sorted_labels, cumsum_percentages, color='red', linestyle='--', marker='o', label='Cumulative Percentage')
    ax2.tick_params(axis='y', colors='red')
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylim(60, 100)

    # Save the plot as a PDF
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

def find_result(model, model_lspr, model_surplus, bounds):
    def f_opt(X):
        quadratic_combinations = list(itertools.combinations_with_replacement(range(len(X)), 2))
        features = np.array(X)[np.newaxis, :]
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

        return f + penalty + surplus_penalty
    result = differential_evolution(f_opt, bounds)
    return result

def features_prep(results, obj_fun):
    # Features
    features = results.iloc[:, 0:4].values
    quadratic_combinations = list(itertools.combinations_with_replacement(range(features.shape[1]), 2))
    feat_quad = np.array([features[:, i:i+1] * features[:, j:j+1] for i, j in quadratic_combinations])
    feat_quad = np.concatenate(feat_quad, axis=1)
    lpsr_values = results.iloc[:, results.shape[1] - 1].values
    features = np.concatenate((features, feat_quad), axis=1)

    # Fit quadratic regression models for each target
    models = {}
    target_values = results.iloc[:, obj_fun].values
    target_lspr = results.iloc[:, -2].values
    target_surplus = results.iloc[:, -6].values

    # Add a constant term to the features
    features_const = sm.add_constant(features)
    return features_const, target_values, target_lspr, target_surplus, quadratic_combinations

def fit_the_model(features_const, target_values, target_lspr, target_surplus):
    model = sm.OLS(target_values, features_const).fit()
    model_lspr = sm.OLS(target_lspr, features_const).fit()
    model_surplus = sm.OLS(target_surplus, features_const).fit()
    return model, model_lspr, model_surplus



# bounds = [(-1,1),(-1,1),(-1,1),(-1,1)]

def denormalize_value(normalized_value, original_min, original_max):
    return original_min + (original_max - original_min) * (normalized_value + 1) / 2

def denormalize(normalized_values, ranges):
    denormalized_values = []
    for i, value in enumerate(normalized_values):
        denormalized_values.append(denormalize_value(value, ranges[i][0], ranges[i][1]))
    return denormalized_values

# # Given ranges
# ranges = [(16, 48), (3, 23), (6, 30), (8, 61)]
#
# # Denormalize
# denormalized_values = denormalize(result.x, ranges)
# print(denormalized_values)
# plot_pareto_chart(model, ['PV', 'BESS', 'rSOC', 'HSS'], quadratic_combinations)




## Check the linear regression
# import seaborn as sns


# Predictions for the current target
# predictions = model.predict(features_const)

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
