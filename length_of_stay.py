import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metricss

data = pd.read_csv("Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv", low_memory=False)


# For simplicity, let's drop rows with missing values and drop non-essential columns
# Drop Unuseful Columns
unuseful_columns = ['Health Service Area', 'Hospital County', 'Operating Certificate Number', 
                    'Facility Name', 'Discharge Year', 'CCS Diagnosis Description', 
                    'CCS Procedure Description', 'APR DRG Description', 'APR MDC Description', 
                    'APR Severity of Illness Description', 'APR Medical Surgical Description', 
                    'Attending Provider License Number', 'Operating Provider License Number', 
                    'Other Provider License Number','Payment Typology 1','Payment Typology 2','Payment Typology 3',
                    'Total Costs']
data.drop(columns=unuseful_columns, inplace=True)

# For simplicity, let's drop rows with missing values and drop non-essential columns
# Drop Unuseful Columns
unuseful_columns = ['Zip Code - 3 digits']
data.drop(columns=unuseful_columns, inplace=True)

data['Total Charges'] = data['Total Charges'].apply(lambda x: str(x).replace('$',''))
data['Total Charges'] = pd.to_numeric(data['Total Charges'])

# Impute missing values in 'Facility Id' column based on the proportion of occurrence
facility_id_counts = data['Facility Id'].value_counts(normalize=True)
missing_indices = data['Facility Id'].isnull()
data.loc[missing_indices, 'Facility Id'] = np.random.choice(facility_id_counts.index, size=missing_indices.sum(), p=facility_id_counts.values)


# Impute missing values in 'APR Risk of Mortality' column based on the proportion of occurrence
facility_id_counts = data['APR Risk of Mortality'].value_counts(normalize=True)
missing_indices = data['APR Risk of Mortality'].isnull()
data.loc[missing_indices, 'APR Risk of Mortality'] = np.random.choice(facility_id_counts.index, size=missing_indices.sum(), p=facility_id_counts.values)


# Replace '120+' with a maximum value (e.g., 120)
data['Length of Stay'] = data['Length of Stay'].replace('120 +', '120')

# Convert 'Length of Stay' column to numeric type
data['Length of Stay'] = pd.to_numeric(data['Length of Stay'])


x= data.drop(columns=['Length of Stay','Race','Ethnicity','Patient Disposition','CCS Procedure Code','Birth Weight','Abortion Edit Indicator','Emergency Department Indicator'])
y = data['Length of Stay']

# Encode categorical variables if necessary
label_encoders = {}
for column in x.select_dtypes(include='object').columns:
    label_encoders[column] = LabelEncoder()
    x[column] = label_encoders[column].fit_transform(x[column])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_pred_test  = lr_model.predict(x_test)

import joblib
joblib.dump(lr_model,'lr.pkl')