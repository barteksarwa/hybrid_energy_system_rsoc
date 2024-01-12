import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assuming designs_modified_df is your DataFrame with features and targets
# Path to the 'designs_modified' file
designs_modified_file_path = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\designs_with_losses.xlsx'

# Load the 'designs_modified' file
designs_modified_df = pd.read_excel(designs_modified_file_path)

# Use the correct column indices (0-based)
features = designs_modified_df.iloc[:, [0,1,2,3]]  
target_loss = designs_modified_df.iloc[:, 4]  
target_deficit = designs_modified_df.iloc[:, 5]
target_cost = designs_modified_df.iloc[:, 6]

# Train the linear regression models
model_loss = LinearRegression()
model_loss.fit(features, target_loss)

model_def = LinearRegression()
model_def.fit(features, target_deficit)

model_cost = LinearRegression()
model_cost.fit(features, target_cost)

# Display the magnitudes of the coefficients in bar charts
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Model Loss
coefficients_loss = model_loss.coef_
axes[0].barh(designs_modified_df.columns[8:13], np.abs(coefficients_loss))
axes[0].set_xlabel('Magnitude of Coefficients')
axes[0].set_ylabel('Feature Names')
axes[0].set_title('Magnitude of Coefficients in Linear Regression Model (Energy Loss)')

# Model Deficit
coefficients_def = model_def.coef_
axes[1].barh(designs_modified_df.columns[8:13], np.abs(coefficients_def))
axes[1].set_xlabel('Magnitude of Coefficients')
axes[1].set_ylabel('Feature Names')
axes[1].set_title('Magnitude of Coefficients in Linear Regression Model (Energy Deficit)')

# Model Cost
coefficients_cost = model_cost.coef_
axes[2].barh(designs_modified_df.columns[8:13], np.abs(coefficients_cost))
axes[2].set_xlabel('Magnitude of Coefficients')
axes[2].set_ylabel('Feature Names')
axes[2].set_title('Magnitude of Coefficients in Linear Regression Model (System Cost)')

plt.tight_layout()
plt.show()