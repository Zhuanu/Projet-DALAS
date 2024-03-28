import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------

file_path = 'data/housedata.csv'
df = pd.read_csv(file_path)

X = df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors"]]
y = df["price"]

# date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,city,statezip,country

# -------------------------------------

models = [
    LinearRegression(),
    Ridge(),
    DecisionTreeRegressor(),
    RandomForestRegressor()
]

models_name = [
    "LinearRegression",
    "Ridge",
    "DecisionTreeRegressor",
    "RandomForestRegressor"
]

for model, name in zip(models, models_name) :
    print("--------------------------")
    print(name)
    model.fit(X, y)
    print(model.predict(X))

    ## Trouver un moyen de plot
    # plt.scatter(X, y, color='blue', label='Training Data')
    # plt.plot(X, model.predict(X), color='green', label='Regression Line')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.title(name)
    # plt.legend()
    # plt.show()

# -------------------------------------

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Generate some sample data
# np.random.seed(0)
# X = 2 * np.random.rand(100, 1)  # Generate 100 random numbers between 0 and 2
# y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3*X + some noise

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the Linear Regression model
# model = LinearRegression()

# # Train the model on the training data
# model.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = model.predict(X_test)

# # Calculate Mean Squared Error (MSE) to evaluate the model's performance
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)

# # Plot the training data, testing data, and the regression line
# import matplotlib.pyplot as plt

# plt.scatter(X_train, y_train, color='blue', label='Training Data')
# plt.scatter(X_test, y_test, color='red', label='Testing Data')
# plt.plot(X_test, y_pred, color='green', label='Regression Line')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Linear Regression Example')
# plt.legend()
# plt.show()