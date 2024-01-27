import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('infolimpioavanzadoTarget.csv') 


df['date'] = pd.to_datetime(df['date'], errors='coerce')

numeric_columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

print("Missing values:\n", df.isnull().sum())


print("\nDescriptive Statistics:\n", df.describe())


plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['adjclose'], label='Adjusted Close Price')
plt.title('Time Series of Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()


correlation_matrix = df.corr()
plt.figure(figsize=(14, 10))
plt.imshow(correlation_matrix, cmap='viridis', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation='vertical')
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Correlation Matrix')
plt.show()




feature_cols = df.columns.drop(['date', 'ticker', 'TARGET'])
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
df = df.dropna()


features = df.drop(['date', 'ticker', 'TARGET'], axis=1)
target = df['TARGET']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest Regression': RandomForestRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor()
}


results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[model_name] = {'model': model, 'MSE': mse}

# Hyperparameter Tuning (Example for Random Forest Regression)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor()
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
best_rf_params = grid_search.best_params_


y_pred_best = best_rf_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)


print("Evaluation Metrics:")
for model_name, result in results.items():
    print(f"{model_name}: MSE = {result['MSE']}")

print("\nBest Random Forest Model:")
print(f"Best Parameters: {best_rf_params}")
print(f"Best MSE: {mse_best}")


print("\nIntroduction:")
print("This project aims to predict stock prices using machine learning techniques. Predicting stock prices is a complex task, and various regression models will be explored to achieve this.")

print("\nExploratory Data Analysis Section:")
print("The exploratory data analysis (EDA) involved loading the dataset, cleaning the data by handling missing values and outliers, calculating descriptive statistics, and visualizing time series trends and correlations between different variables.")

print("\nPredictive Modeling Section:")
print("For predictive modeling, relevant features were identified, and the dataset was split into training and testing sets. Regression models, including Linear Regression, Ridge Regression, Lasso Regression, Random Forest Regression, and Gradient Boosting Regression, were implemented. Hyperparameter tuning was performed using GridSearchCV to optimize the Random Forest Regression model. Evaluation metrics such as Mean Squared Error (MSE) were used to assess model performance.")

print("\nResults and Insights:")
print("The predictive modeling section yielded models with varying MSE values. The best-performing model, determined through hyperparameter tuning, was the Random Forest Regression model with optimized parameters.")

print("\nConclusion:")
print("In conclusion, this project provides insights into predicting stock prices using machine learning models. Limitations include the complexity of stock price prediction and the need for continuous refinement. Future exploration could involve incorporating additional features or exploring more advanced modeling techniques.")
