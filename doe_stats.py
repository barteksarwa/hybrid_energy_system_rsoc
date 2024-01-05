import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

BREAKPOINT = 90

# Assuming designs_modified_df is your DataFrame with features and targets
# Path to the 'designs_modified' file
designs_modified_file_path = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\designs_modified.xlsx'

# Load the 'designs_modified' file
designs_modified_df = pd.read_excel(designs_modified_file_path)

# Define targets
targets = {
    'Energy Loss': designs_modified_df.iloc[:, 5],
    'Energy Deficit': designs_modified_df.iloc[:, 6],
    'Cost': designs_modified_df.iloc[:, 7]
}

# Features
features = designs_modified_df.iloc[:, [8, 9, 10, 11, 12]]
feature_labels = ['PV', 'B', 'EC', 'FC', 'tank']


# Add quadratic terms for each feature
quadratic_combinations = list(itertools.combinations_with_replacement(range(len(features.columns)), 2))
quadratic_terms = [f"{feature_labels[i]}*{feature_labels[j]}" for i, j in quadratic_combinations]
features_quadratic = pd.concat([features] + [features.iloc[:, i] * features.iloc[:, j] for i, j in quadratic_combinations], axis=1)
features_quadratic.columns = feature_labels + quadratic_terms

# Fit quadratic regression models for each target
models = {}
for target_name, target_values in targets.items():
    # Add a constant term to the features
    features_const = sm.add_constant(features_quadratic)

    # Fit the model
    model = sm.OLS(target_values, features_const).fit()
    models[target_name] = model

    # Display bar chart for coefficients
    coefficients = model.params[1:]  # Exclude the intercept
    sorted_coefficients = coefficients.abs().sort_values(ascending=False)

    # Bar labels
    bar_labels = feature_labels + quadratic_terms[:len(coefficients) - len(feature_labels)]

    # Rearrange bar labels based on sorted coefficients
    sorted_labels = [label for _, label in sorted(zip(coefficients.index, bar_labels),
                                                 key=lambda x: sorted_coefficients.index.get_loc(x[0]))]

    plt.figure(figsize=(10, 5))

    # Bar plot
    ax1 = plt.gca()
    ax1.bar(sorted_labels, sorted_coefficients, label='Magnitude of Coefficients')

    # Calculate cumulative sum and normalize to get percentages
    cumsum_percentages = np.cumsum(sorted_coefficients) / sorted_coefficients.sum() * 100

    # Line plot for cumulative percentages on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(sorted_labels, cumsum_percentages, color='red', linestyle='--', marker='o', label='Cumulative Percentage')

    # Find the index where cumulative percentage exceeds 80%
    exceed_index = np.argmax(cumsum_percentages > BREAKPOINT)
    
    # If all values are below the breakpoint, set the index to the last one
    if exceed_index == 0:
        exceed_index = len(sorted_labels) - 1
    
    # Mark the point on the line plot with a vertical line
    ax2.axvline(x=exceed_index, color='green', linestyle='-', label=f"Exceeds {BREAKPOINT}%")

    ax1.set_xlabel('Features')
    ax1.set_ylabel('Magnitude of Coefficients')
    ax2.set_ylabel('Cumulative Percentage')
    plt.title(f'Magnitude of Coefficients and Cumulative Percentage (System {target_name})')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.show()

    # Display summary statistics
    print(f"Summary Statistics for Quadratic Regression Model ({target_name}):\n\n{model.summary()}\n\n")
    
    model_preds = {}
    
    # Predictions for the current target
    predictions = model.predict(features_const)
    model_preds[target_name] = predictions

    print(f"Predictions for {target_name}:\n{predictions}\n")