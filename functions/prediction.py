import pandas as pd
import pickle


def read_data(filename):
    model = pickle.load(open(filename, 'rb'))
    return model

data_path = './data/'
rf_regressor = read_data(data_path + 'random_forest_regression_model.pkl')

def make_predictions(year, present_price, kms_driven, owner, fuel_type, seller_type, transmission_type):
    
    fuel_type_diesel, fuel_type_petrol = 0, 0
    if(fuel_type == 'Petrol'):
            fuel_type_petrol = 1
            fuel_type_diesel = 0
    else:
        fuel_type_petrol = 0
        fuel_type_diesel = 1

    year = 2021 - year
    seller_type = 1 if(seller_type == 'Individual') else 0    
    transmission_type = 1 if(transmission_type == 'Manual') else 0
    prediction = rf_regressor.predict([[present_price, kms_driven, owner, year, fuel_type_diesel, fuel_type_petrol, seller_type,transmission_type]])
    output = round(prediction[0], 2)
    if output < 0:
        return "You cannot sell this vehicle!"
    else:
        return "You can sell the vehicle at Rs. {} lakhs".format(output)
