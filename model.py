import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df = pd.read_csv('Car_sales.csv')
    #Data is splited into train and test category
    train, test = data_split(df, 0.2)

    # Split the input in X factor
    x_train = train[['Price_in_thousands','Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency']].to_numpy()
    x_test = test[['Price_in_thousands','Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency']].to_numpy()

    # Split the output in Y factor
    y_train = train[['Power_perf_factor']].to_numpy().reshape(126,-1)
    y_test = test[['Power_perf_factor']].to_numpy().reshape(31,-1)

    # Linear model Build
    reg = linear_model.LinearRegression()

    # Provide Data to Model
    reg.fit(x_train, y_train)


    modelPrediction = reg.predict(x_test)
    
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, modelPrediction))

    ##The coefficient of determination: 1 is perfect prediction

    print('Coefficient of determination: %.2f'
        % r2_score(y_test, modelPrediction))


    # Predict the Value from the model
    modelPrediction = reg.predict([[39.605,   4   , 620   , 173   ,  78.5  , 200.1  ,   5.9  ,11.2  ,  11   ]])
    print("Result",modelPrediction)


