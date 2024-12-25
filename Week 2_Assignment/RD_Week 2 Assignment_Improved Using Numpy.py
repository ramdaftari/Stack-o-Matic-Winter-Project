import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr: float = 0.01, n_iters: int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        num_samples, num_features = X.shape
        self.weights = np.random.rand(num_features)
        self.bias = 0

        for i in range(self.n_iters):
            Y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X.T, Y_pred - Y)
            db = (1 / num_samples) * np.sum(Y_pred - Y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

df = pd.read_csv(r'C:\Users\Ram\Downloads\extended_salary_data.csv', header=0)
df.rename(columns={'YearsExperience': 'Years of Experience'}, inplace=True)
print(df)

plt.figure()
plt.scatter(df['Years of Experience'], df['Salary'], color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Dataframe')
plt.show(block=False)

np.random.seed(42)
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
index = int(len(df) * 0.8)
X_train, Y_train = df['Years of Experience'][:index].values.reshape(-1, 1), df['Salary'][:index]
X_test, Y_test = df['Years of Experience'][index:].values.reshape(-1, 1), df['Salary'][index:]

model = LinearRegression(lr=0.01, n_iters=100000)
model.fit(X_train, Y_train)

Y_predicted = model.predict(X_test)
mse = np.mean((Y_predicted - Y_test.values) ** 2)
print(f'Mean Squared Error: {mse:.3f}')

fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('off')
comp_table = ax.table(
    cellText=[[f"{actual:.3f}", f"{predicted:.3f}"] for actual, predicted in zip(Y_test.values.flatten(), Y_predicted)],
    colLabels=['Actual', 'Predicted'],
    loc='center'
)
for (i, j), cell in comp_table.get_celld().items():
    cell.set_text_props(ha='center', va='center')
plt.show(block=False)

plt.figure()
X_range = np.linspace(df['Years of Experience'].min(), df['Years of Experience'].max(), 100).reshape(-1, 1)
Y_pred_line = model.predict(X_range)
plt.scatter(df['Years of Experience'], df['Salary'], color='blue', label='Actual Data')
plt.plot(X_range, Y_pred_line, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
