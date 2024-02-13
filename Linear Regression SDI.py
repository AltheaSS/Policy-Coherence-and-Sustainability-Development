#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt



# Modeling SDI on PCDI
# Load the datasets from Excel files
org_PCDI = pd.read_excel('/Users/noira/Desktop/PCDI-2019.xlsx', na_values=['?', '-', 'n/a'])
org_SDI = pd.read_excel('/Users/noira/Desktop/web_scrap_SDI_2019.xlsx', na_values=['?', '-', 'n/a'])
org_BTI = pd.read_excel('/Users/noira/Desktop/BTI 2018.xlsx', na_values=['?', '-', 'n/a'])


# Merge the datasets on the 'Country' column
merged_org_PCDI_SDI = pd.merge(org_PCDI, org_SDI, left_on = 'COUNTRIES', right_on = 'Country', how = 'right')

merged_org_PCDI_SDI = merged_org_PCDI_SDI.dropna()

X = merged_org_PCDI_SDI[['PCSDI']].values.reshape(-1,1)
y = merged_org_PCDI_SDI['SDI'].values

correlation = merged_org_PCDI_SDI['PCSDI'].corr(merged_org_PCDI_SDI['SDI'])
print(f'Pearson correlation coefficient between PCDI and SDI: {correlation}')




# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating linear regression model
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)

# Making prediction
predictions = model.predict(X_test)

print(f"Coefficient: {model.coef_}, Intercept: {model.intercept_}")


r_squared = r2_score(y_test, predictions)
print("R-squared:", r_squared)


mse = mean_squared_error(y_test, predictions)
rmse = sqrt(mse)
#print("RMSE:", rmse)

org_BTI.columns


# In[100]:


# Modeling SDI on BTI indices


# rename BTI country column
# Example code to handle the non-numeric values

# Replace '-' with NaN and then convert column to numeric, assuming org_BTI and org_SDI are your original DataFrames

org_BTI.columns = org_BTI.columns.str.lstrip()

org_BTI = org_BTI.rename(columns = {org_BTI.columns[0]: 'Country'})
org_BTI['S | Status Index'] = pd.to_numeric(org_BTI['S | Status Index'], errors='coerce')
org_BTI['Q3 | Rule of Law'] = pd.to_numeric(org_BTI['Q3 | Rule of Law'], errors='coerce')
org_SDI['SDI'] = pd.to_numeric(org_SDI['SDI'], errors='coerce')

# Now, merge the data again
merged_org_SDI_BTI = pd.merge(org_BTI, org_SDI, on = 'Country', how='left')

# You might want to handle missing values (NaNs) after conversion. Here's an example of filling NaNs with the mean
merged_org_SDI_BTI['Q3 | Rule of Law'].fillna(merged_org_SDI_BTI['S | Status Index'].mean(), inplace=True)
merged_org_SDI_BTI['SDI'].fillna(merged_org_SDI_BTI['SDI'].mean(), inplace=True)

# Proceed with your original code here
X = merged_org_SDI_BTI['Q3 | Rule of Law'].values.reshape(-1, 1)
y = merged_org_SDI_BTI['SDI'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

r_squared = r2_score(y_test, predictions)
print("R-squared:", r_squared)


mse = mean_squared_error(y_test, predictions)
rmse = sqrt(mse)
print("RMSE:", rmse)

BTI_SDI_lin_corr = {}


merged_org_SDI_BTI.iloc[:, 106]


# In[108]:


# Assuming merged_org_SDI_BTI is your DataFrame and 'SDI' is the target column

# Step 1: Drop columns that only contain NaN values
merged_org_SDI_BTI.dropna(axis=1, how='all', inplace=True)

# Step 2: Replace NaN values in numeric columns with their mean
numeric_columns = merged_org_SDI_BTI.select_dtypes(include=['number']).columns.drop('SDI')  # Excluding 'SDI'
for column in numeric_columns:
    merged_org_SDI_BTI[column].fillna(merged_org_SDI_BTI[column].mean(), inplace=True)

# Step 3: Verify that there are no NaNs left in the dataset
if merged_org_SDI_BTI.isnull().any().any():
    print("Warning: NaN values are still present in the dataset.")
else:
    print("No NaN values found in the dataset.")

# Initialize a dictionary to store R-squared and RMSE values
BTI_SDI_lin_corr = {}

# Loop over numeric columns to perform linear regression against 'SDI'
for column in numeric_columns:
    X = merged_org_SDI_BTI[[column]].values  # Ensure X is a DataFrame slice
    y = merged_org_SDI_BTI['SDI'].values

    # Splitting dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions and evaluating the model
    predictions = model.predict(X_test)
    r_squared = r2_score(y_test, predictions)
    rmse = sqrt(mean_squared_error(y_test, predictions))

    # Store the R-squared value and RMSE
    BTI_SDI_lin_corr[column] = {'R_squared': r_squared, 'RMSE': rmse}

# Print or analyze the R-squared and RMSE values
for column, metrics in BTI_SDI_lin_corr.items():
    print(f"{column}: R-squared = {metrics['R_squared']}, RMSE = {metrics['RMSE']}")



    


# In[ ]:




