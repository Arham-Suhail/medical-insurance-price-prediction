import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request

#url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
file_name = 'medical_insurance.csv'
#urllib.request.urlretrieve(url,file_name)
df=pd.read_csv(file_name,header=None)
headers=["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns=headers
#print(df.head())

df.replace('?',np.nan, inplace=True)

#print("\nMissing Values:")
#print(df.isnull().sum())

#print("\nData Types:")
#print(df.dtypes)

# Age ko numeric karo aur missing ko mean se fill karo
mean_age = df['age'].astype('float').mean(axis=0)
df['age'] = df['age'].fillna(mean_age)

# Smoker ko numeric karo aur missing ko mode (most frequent) se fill karo
is_smoker = df['smoker'].value_counts().idxmax()
df['smoker'] = df['smoker'].fillna(is_smoker)

# Dtypes update karo
df[["age", "smoker"]] = df[["age", "smoker"]].astype("int")

# Charges ko round karo
df[["charges"]] = np.round(df[["charges"]], 2)

#print(df.info())
#print(df.head())

print("=== Correlation with Charges ===")
print(df.corr()['charges'].sort_values(ascending=False))

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
plt.title("Correlation with Charges")
#plt.show()

plt.figure(figsize=(10,6))
sns.regplot(x='bmi',y='charges',data=df)
plt.ylim(0,)
plt.title("BMI VS CHARGES")
#plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x="smoker",y="charges",data=df)
plt.title("Smoker vs Charges")
#plt.show()

plt.figure(figsize=(10,6))
sns.regplot(x='age',y='charges',data=df)
plt.ylim(0,)
plt.title("Age vs Charges")
#plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
lm = LinearRegression()

X = df[['smoker']]
Y = df['charges']

lm.fit(X,Y)
Yhat = lm.predict(X)

print("=== SLR: Smoker → Charges ===")
print("Intercept (a):", lm.intercept_)
print("Slope (b):", lm.coef_)
print("R-square:", lm.score(X, Y))
print("MSE:", mean_squared_error(Y, Yhat))

Z = df[['smoker', 'age', 'bmi']]

lm.fit(Z, Y)
Yhat_mlr = lm.predict(Z)

print("\n=== MLR: Smoker + Age + BMI → Charges ===")
print("Intercept (a):", lm.intercept_)
print("Coefficients:", lm.coef_)
print("R-square:", lm.score(Z, Y))
print("MSE:", mean_squared_error(Y, Yhat_mlr))

plt.figure(figsize=(10, 6))
sns.kdeplot(Y, label="Actual Charges", color="r", fill=True)
sns.kdeplot(Yhat,label="SLR Predicted Charges", color="g", fill=True)
sns.kdeplot(Yhat_mlr, label="MLR Predicted Charges", color="b", fill=True)
plt.title("Actual vs SLR vs MLR")
plt.xlabel("Charges")
plt.legend()
#plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

#Train test split
Z = df[['smoker','age','bmi']]
Y = df['charges']
Z_train,Z_test,Y_train,Y_test = train_test_split(Z,Y,test_size=0.2, random_state=1)
lm.fit(Z_train,Y_train)
print("=== MLR on Test Data ===")
print("R-square:", lm.score(Z_test, Y_test))

RidgeModel = Ridge(alpha=1)
RidgeModel.fit(Z_train,Y_train)
print("\n=== Ridge Regression (alpha=1) ===")
print("R-square:", RidgeModel.score(Z_test, Y_test))


from sklearn.model_selection import GridSearchCV

parameters = [{'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

RR = Ridge()
Grid = GridSearchCV(RR, parameters, cv=4)
Grid.fit(Z_train, Y_train)

BestRR = Grid.best_estimator_
print("\n=== Best Ridge Model ===")
print("Best alpha:", Grid.best_params_)
print("R-square on test data:", BestRR.score(Z_test, Y_test))


alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
r2_scores = []

for a in alphas:
    RidgeModel = Ridge(alpha=a)
    RidgeModel.fit(Z_train, Y_train)
    r2_scores.append(RidgeModel.score(Z_test, Y_test))

plt.figure(figsize=(10, 6))
plt.plot(alphas, r2_scores, marker='o')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R² (Test Data)')
plt.title('Ridge Regression: Alpha vs R²')
plt.show()




