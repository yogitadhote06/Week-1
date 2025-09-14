#PROJECT TITLE: Carbon Dioxide Emmisions Prediction Rate

#PROBLEM STATEMENT: Accurate prediction of CO₂ emission rates is critical for informing climate policy, optimizing energy strategies, and enhancing disaster preparedness. However, existing models often struggle with integrating real-time data, accounting for socio-economic variables, and adapting to regional disparities. This project aims to develop a robust, data-driven model to forecast CO₂ emission rates with higher precision, enabling proactive climate action and improved disaster management planning.

#DESCRIPTION: This Project uses a dataset consisting of Carbon Dioxide emitted by different vehicles in a city (like Cannada).the model will focus on predicting CO2 emissions prediction rate so that rate could be taken into account for further processing for reducing CO2 emissions.

#Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
#Load the dataset
df=pd.read_csv('C:/Users/HP/Downloads/CO2 Emissions_Canada.csv')
#display first few rows
print(df.head())
print(df.describe())
print("\n Missing Values:")
print(df.isnull().sum())
# Rows and Cols
df.shape
# Cols
df.columns
#Univariate Analysis
sns.countplot(x='CO2 Emissions(g/km)', data=df)
plt.title('CO2 Emissions(g/km)')
plt.show()

num_cols = ['Fuel Consumption City (L/100 km)',
       'Fuel Consumption Hwy (L/100 km)']
fig, axes = plt.subplots(1, 2, figsize=(10,3))
for ax, col in zip(axes, num_cols):
    sns.histplot(df[col], kde=True, ax = ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# Bivariate analysis
fig, axes = plt.subplots(1, 2, figsize=(10,6))
sns.boxplot(x = 'CO2 Emissions(g/km)', y = 'Fuel Consumption City (L/100 km)', data = df, ax=axes[0]).set_title('CO2 Emissions(g/km) vs Fuel Consumption City (L/100 km)')
#sns.countplot(x = 'Engine Size(L)', hue = 'CO2 Emissions(g/km)', data = df, ax=axes[1]).set_title('CO2 Emissions(g/km) vs Engine Size(L)')
sns.countplot(x = 'Fuel Type', hue = 'CO2 Emissions(g/km)', data = df, ax=axes[1]).set_title('CO2 Emissions(g/km)')
plt.tight_layout()
plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=(5,3))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Pairplot for numerical features colored by CO2 Emissions(g/km)
sns.pairplot(df, vars=num_cols, hue='CO2 Emissions(g/km)')
plt.show()

# Data Preprocessing
le = LabelEncoder()
categorical_cols = ['Make', 'Model', 'Vehicle Class', 
       'Transmission', 'Fuel Type']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature and target selection
X = df.drop('CO2 Emissions(g/km)', axis =1)
y = df['CO2 Emissions(g/km)']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape

X_test.shape

# scale num features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_train

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.linear_model import LinearRegression

lr= LinearRegression() 

lr.fit(X_train, y_train)

# Predict using the model on the test data

y_train_pred = lr.predict(X_train)
y_pred = lr.predict(X_test)

sns.regplot(x=y_test, y=y_pred, ci=None)

plt.xlabel('y Test')
plt.ylabel('y Pred')

plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_val(y_train, y_train_pred, y_test, y_pred, i): 
    
    scores = {
    i+"_train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    i+"_test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}
    }
    
    return pd.DataFrame(scores)

slr_score = train_val(y_train, y_train_pred, y_test, y_pred, 'linear')
slr_score

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
#Evaluate both models
evaluate_model(rf, X_test, y_test, "Random Forest")

#Save the model
joblib.dump(lr, 'lr_model.pkl')

#Save the scaler
joblib.dump(scaler, 'scaler.pkl')
