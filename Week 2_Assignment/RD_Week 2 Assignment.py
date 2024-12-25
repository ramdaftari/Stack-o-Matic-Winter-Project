import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Exploratory Data Analysis
df = pd.read_csv(r'C:\Users\Ram\Downloads\extended_salary_data.csv', header=0)
df.rename(columns={'YearsExperience': 'Years of Experience'}, inplace=True)
print(df)

# Scatter plot of the data
plt.figure()
plt.scatter(df['Years of Experience'], df['Salary'], color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Dataframe')
plt.show(block=False)

print("Mean Salary :", f"{df.loc[:,'Salary'].mean():.3f}")
print("Median Salary :", f"{df.loc[:,'Salary'].median():.3f}")
print("Mean Years of Experience :", f"{df.loc[:,'Years of Experience'].mean():.3f}")
print("Median Years of Experience :", f"{df.loc[:,'Years of Experience'].median():.3f}")

# Build a Linear Regression Model
X = df[['Years of Experience']]
Y = df[['Salary']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
lr = LinearRegression()
lr.fit(X_train, Y_train)
accuracy = lr.score(X_test, Y_test) * 100
print("Model Accuracy :", f"{accuracy:.3f}")
print(f"Slope of Linear Model: {lr.coef_[0][0]:.3f}")
print(f"Y-Intercept of Linear Model: {lr.intercept_[0]:.3f}")

# Evaluate the Model
Y_pred = (lr.predict(X_test)).flatten()
Y_test_flatten = (Y_test.values).flatten()
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse:.3f}")
mae = mean_absolute_error(Y_test, Y_pred)
print(f"Mean Absolute Error : {mae:.3f}")

# Table of Actual vs Predicted
fig, ax = plt.subplots(figsize=(6, 3))  # No extra plt.figure() here
ax.axis('off')
comp_table = ax.table(
    cellText=[[f"{actual:.3f}", f"{predicted:.3f}"] for actual, predicted in zip(Y_test_flatten, Y_pred)],
    colLabels=['Actual', 'Predicted'],
    loc='center'
)
for (i, j), cell in comp_table.get_celld().items():
    cell.set_text_props(ha='center', va='center')
plt.show(block=False)

# Regression line plot
plt.figure()
X_range = np.linspace(df['Years of Experience'].min(), df['Years of Experience'].max(), 100).reshape(-1, 1)
Y_pred_line = lr.predict(X_range)
plt.scatter(df['Years of Experience'], df['Salary'], color='blue', label='Actual Data')
plt.plot(X_range, Y_pred_line, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
