import itertools
import pandas as pd
import statsmodels.api as sm

# Provided data
data = {
    'PV': [-1],
    'B': [0],
    'EC': [-1],
    'FC': [0],
    'tank': [0],
    'Energy Deficit': [151.2489553]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Features
features = df[['PV', 'B', 'EC', 'FC', 'tank']]
feature_labels = ['PV', 'B', 'EC', 'FC', 'tank']

# Add quadratic terms for each feature
quadratic_combinations = list(itertools.combinations_with_replacement(range(len(features.columns)), 2))
quadratic_terms = [f"{feature_labels[i]}*{feature_labels[j]}" for i, j in quadratic_combinations]
features_quadratic = pd.concat([features] + [features.iloc[:, i] * features.iloc[:, j] for i, j in quadratic_combinations], axis=1)
features_quadratic.columns = feature_labels + quadratic_terms

# Add a constant term to the features
features_const = sm.add_constant(features_quadratic)

# Fit the OLS model
ols_model = sm.OLS(df['Energy Deficit'], features_const).fit()

# Use the predict method to get the predicted value for 'Energy Deficit'
predicted_energy_deficit = ols_model.predict(features_const)

# Display the OLS model summary
print("Summary Statistics for Quadratic Regression Model (Energy Deficit):\n")
print(ols_model.summary())

# Display the provided data and the predicted value for 'Energy Deficit'
print("\nProvided Data:")
print(df)
print("\nPredicted Energy Deficit:")
print(predicted_energy_deficit.values[0])