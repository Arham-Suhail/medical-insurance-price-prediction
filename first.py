import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# SETUP: Download and load the data
# ============================================================
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"  # replace with your actual URL if different
file_name = "usedcars.csv"
urllib.request.urlretrieve(url, file_name)

df = pd.read_csv(file_name)
print("=== Data Preview ===")
print(df.head())


# ============================================================
# SECTION 1: Simple Linear Regression (SLR)
# ============================================================
lm = LinearRegression()

X = df[['highway-mpg']]
Y = df['price']
lm.fit(X, Y)

Yhat = lm.predict(X)
print("\n=== Simple Linear Regression ===")
print("First 5 predictions:", Yhat[0:5])
print("Intercept (a):", lm.intercept_)
print("Slope (b):", lm.coef_)


# ============================================================
# SECTION 2: Multiple Linear Regression (MLR)
# ============================================================
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])

print("\n=== Multiple Linear Regression ===")
print("Intercept (a):", lm.intercept_)
print("Coefficients (b1, b2, b3, b4):", lm.coef_)


# ============================================================
# SECTION 3: Model Evaluation Using Visualization
# ============================================================

# --- Regression plot: highway-mpg vs price ---
plt.figure(figsize=(12, 10))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.title("Regression Plot: highway-mpg vs price")
plt.show()

# --- Regression plot: peak-rpm vs price ---
plt.figure(figsize=(12, 10))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.title("Regression Plot: peak-rpm vs price")
plt.show()

# --- Correlation check: which is more strongly correlated with price? ---
print("\n=== Correlation: peak-rpm vs highway-mpg vs price ===")
print(df[["peak-rpm", "highway-mpg", "price"]].corr())

# --- Residual plot ---
plt.figure(figsize=(12, 10))
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.title("Residual Plot: highway-mpg vs price")
plt.show()

# --- Distribution plot: Actual vs Fitted (for MLR) ---
Y_hat = lm.predict(Z)
plt.figure(figsize=(12, 10))
sns.kdeplot(df['price'], label="Actual Value", color="r", fill=True)
sns.kdeplot(Y_hat, label="Fitted Values", color="b", fill=True)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()


# ============================================================
# SECTION 4: Polynomial Regression
# ============================================================
x = df['highway-mpg']
y = df['price']

# Fit a 3rd order (cubic) polynomial
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print("\n=== Polynomial (cubic) function ===")
print(p)

def plot_polynomial(model, independent_variable, dependent_variable, name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit for Price ~ ' + name)
    plt.xlabel(name)
    plt.ylabel('Price of Cars')
    plt.show()

plot_polynomial(p, x, y, 'highway-mpg')


# ============================================================
# SECTION 5: Pipeline (Scale + Polynomial Features + Model)
# ============================================================
Input = [
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression())
]
pipe = Pipeline(Input)

Z = Z.astype(float)
pipe.fit(Z, y)
ypipe = pipe.predict(Z)
print("\n=== Pipeline Predictions (first 4) ===")
print(ypipe[0:4])


# ============================================================
# SECTION 6: Model Evaluation - R-squared and MSE
# ============================================================

# --- Model 1: Simple Linear Regression ---
lm.fit(X, Y)
print("\n=== Model 1: Simple Linear Regression ===")
print("R-square:", lm.score(X, Y))
Yhat = lm.predict(X)
print("MSE:", mean_squared_error(df['price'], Yhat))

# --- Model 2: Multiple Linear Regression ---
lm.fit(Z, df['price'])
print("\n=== Model 2: Multiple Linear Regression ===")
print("R-square:", lm.score(Z, df['price']))
Y_predict_multifit = lm.predict(Z)
print("MSE:", mean_squared_error(df['price'], Y_predict_multifit))

# --- Model 3: Polynomial Fit ---
r_squared = r2_score(y, p(x))
print("\n=== Model 3: Polynomial Fit ===")
print("R-square:", r_squared)
print("MSE:", mean_squared_error(df['price'], p(x)))


# ============================================================
# SECTION 7: Prediction with New Input
# ============================================================
new_input = np.arange(1, 100, 1).reshape(-1, 1)

lm.fit(X, Y)
yhat_new = lm.predict(new_input)
print("\n=== Predictions for new input (first 5) ===")
print(yhat_new[0:5])

plt.plot(new_input, yhat_new)
plt.title("Prediction on New Input Range")
plt.xlabel("highway-mpg (new input)")
plt.ylabel("Predicted Price")
plt.show()


# ============================================================
# CONCLUSION
# ============================================================
# Compare R-squared and MSE across all 3 models:
# - Higher R-squared = better fit
# - Lower MSE = better fit
# Typically, MLR (multiple linear regression) performs best since
# it uses more relevant predictor variables (horsepower, curb-weight,
# engine-size, highway-mpg) instead of just one.