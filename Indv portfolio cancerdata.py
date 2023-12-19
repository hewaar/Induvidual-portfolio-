# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:56:01 2023

@author: hewaa
"""

# merge 2 datasets 
import pandas as pd

# Lees de datasets in
df1 = pd.read_csv('avg-household-size.csv')
df2 = pd.read_csv('cancer_reg.csv')

# Samenvoegen op basis van de gemeenschappelijke variabele "geography"
merged_df = pd.merge(df1, df2, on='geography')

# Bekijk de samengevoegde dataset
print(merged_df.head())

# Optioneel: opslaan van de samengevoegde dataset naar een nieuw bestand
merged_df.to_csv('samengevoegde_dataset.csv', index=False)

# Data exploration 
# general info 
print(merged_df.info())

#statistical summary
print(merged_df.describe())

#correlation matrix
correlation_matrix = merged_df.corr()
print(correlation_matrix)




# Check for missing values 
# Verwijder rijen met ontbrekende waarden
merged_df_cleaned = merged_df.dropna()

# Bekijk de informatie van de schoongemaakte dataset
print(merged_df_cleaned.info())


# get dummies 
# Voer one-hot encoding uit op de categorische variabele "binnedinc"
merged_df_encoded = pd.get_dummies(merged_df_cleaned, columns=['binnedinc'])

# Bekijk de informatie van de dataset na one-hot encoding
print(merged_df_encoded.info())


# exploretive data analysis 
# universal analysis 
# histogram 
import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogrammen van numerieke variabelen
numerical_columns = merged_df_encoded.select_dtypes(include=['float64', 'int64']).columns
merged_df_encoded[numerical_columns].hist(bins=20, figsize=(15, 15))
plt.show()

# boxplots 
# Plot boxplots van numerieke variabelen
plt.figure(figsize=(15, 10))
sns.boxplot(data=merged_df_encoded[numerical_columns])
plt.show()

# Bivariate Analysis
# Plot pairplot voor een subset van variabelen
subset_of_variables = ['target_deathrate', 'medincome', 'pctwhite', 'pctblack', 'pctasian', 'pctmarriedhouseholds']
sns.pairplot(merged_df_encoded[subset_of_variables])
plt.show()

# heatmap 
# Plot een heatmap van de correlatiematrix
correlation_matrix = merged_df_encoded.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Categorical Variables
# Plot countplot voor een categorische variabele
plt.figure(figsize=(10, 6))
sns.countplot(x='binnedinc_(34218.1, 37413.8]', data=merged_df_encoded)
plt.show()


# Model building 
# Selection of independent variables using regression analysis
# Kies variabelen met een significante correlatie
significant_vars = correlation_matrix['target_deathrate'][abs(correlation_matrix['target_deathrate']) > 0.2].index

# Voer een regressieanalyse uit met de geselecteerde variabelen
X_selected = merged_df_encoded[significant_vars]
y = merged_df_encoded['target_deathrate']

X_selected = sm.add_constant(X_selected)  # Voeg een constante toe voor de intercept
model = sm.OLS(y, X_selected).fit()
print(model.summary())

# Strong multicollinearity 
# Random forest
# dataset contains non-numeric columns
# Encode Categorical Variables 
# Assuming X is your feature matrix
X_encoded = pd.get_dummies(X)

# Missing Values 
# Remove rows with missing values
X_no_missing = X_encoded.dropna()
y_no_missing = y.loc[X_no_missing.index]

# Train Random forest regression 
rf_model = RandomForestRegressor()
rf_model.fit(X_no_missing, y_no_missing)


# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importances
importances_df = pd.DataFrame({'Feature': X_no_missing.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Print or visualize the feature importances
print(importances_df)

# Plotting a bar chart for better visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importances')
plt.show()

# Set een threshold voor feature importance
threshold = 0.01  # Je kunt deze waarde aanpassen op basis van je voorkeur

# Filter de features op basis van de threshold
selected_features = importances_df[importances_df['Importance'] > threshold]['Feature']

# Creëer een nieuwe DataFrame met alleen de geselecteerde features
X_selected = X_no_missing[selected_features]

# Splits de data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train het model
from sklearn.linear_model import Ridge

# Creëer een Ridge regressie model
ridge_model = Ridge(alpha=1.0)

# Train het model
ridge_model.fit(X_train, y_train)

# Evalueer het model
from sklearn.metrics import mean_squared_error, r2_score
y_pred = ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')



# Model diagnostics
# Calculate the Residues
residuals = y_test - y_pred

# Calculate the Residual Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf

# Plot de autocorrelatiefunctie van de residuen
plot_acf(residuals, lags=20)  # Pas het aantal lags aan indien nodig
plt.show()

# Plot the Pattern of Residuals
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--', lw=2)  # Voeg een horizontale lijn toe op nul
plt.xlabel('Voorspelde waarden')
plt.ylabel('Residuen')
plt.title('Residu Plot voor Heteroskedasticiteit')
plt.show()

# further investegation 
from statsmodels.stats.diagnostic import het_breuschpagan

# Voeg een constante kolom toe aan de voorspelde waarden
X_test_with_constant = sm.add_constant(X_test)

# Bereken de residuen opnieuw (indien nog niet gedaan)
residuals = y_test - y_pred

# Voer de Breusch-Pagan test uit
_, p_value, _, _ = het_breuschpagan(residuals, X_test_with_constant)
print(f'p-value voor Heteroskedasticiteit: {p_value}')

# QQ-plot 
import matplotlib.pyplot as plt
import scipy.stats as stats

# Bereken de residuen
residuals = y_test - y_pred

# Maak een QQ-plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-plot of Residuals")
plt.show()

# Assess multicollinearity 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Bereid de gegevens voor (zorg ervoor dat je de juiste kolommen hebt geselecteerd)
X_selected = X_train  # Pas dit aan op basis van je geselecteerde variabelen

# Voeg een constante toe aan de onafhankelijke variabelenmatrix
X_with_const = sm.add_constant(X_selected)

# Bereken de VIF voor elke variabele
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

# Toon de VIF-gegevens
print(vif_data)


# K-fold cross-validation
# Modelselectie en evaluatie: K-fold cross-validatie
from sklearn.model_selection import cross_val_score, KFold

# Definieer het aantal vouwen (bijvoorbeeld k=10)
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

# Voer k-fold cross-validatie uit op de bijgewerkte gegevens
cross_val_scores = cross_val_score(ridge_model, X_selected, y, cv=k_fold, scoring='neg_mean_squared_error')

# Bereken en druk het gemiddelde van de Mean Squared Errors over de vouwen uit
mean_mse = -cross_val_scores.mean()
print(f'Gemiddelde Mean Squared Error over {k_fold}-fold cross-validatie: {mean_mse}')










