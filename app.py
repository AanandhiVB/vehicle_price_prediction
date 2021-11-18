import json
import pandas as pd
from flask import Flask, request , make_response , render_template
from flask_restplus import Api, Resource, fields, reqparse, Namespace
from functions import prediction
import requests
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

api = Api(app = app, 
		  version = 0.1, 
		  title = "Car Price Prediction APIs", 
		  description = "APIs that can be used for predicting the selling price for a car based on certain parameters.")

mp_api = api.namespace("price_prediction", description="Car Price Prediction APIs")

carprice_model = api.model('CarPriceModel', 
		  {"year": fields.Integer(required=True, default=2016, example=2016, description="Year"),
           "present_price": fields.Float(required=True, default=20.54, example=20.54, description="What is the showroom price? (In lakhs)"),
           "owner": fields.Integer(required=True, default=0, example=0, description="How many owners previously owned the car(0 or 1 or 3) ?"),
           "kms_driven": fields.Integer(required=True, default=2459, example=2459, description="How many kilometers driven?"),
           "fuel_type": fields.String(required=True, description="Fuel Type", default="Petrol", example="Petrol"),
           "seller_type": fields.String(required=True, description="Seller Type", default="Dealer", example="Dealer"),
           "transmission_type": fields.String(required=True, description="Transmission Type", default="Manual", example="Manual"),
           })

@api.route('/api/swagger.json')
class SwaggerJson(Resource):
    @staticmethod
    def get():
        return api.__schema__

@api.route('/api/')
class SwaggerUi(Resource):
    @staticmethod
    def get():
        specs_url = "./swagger.json"
        return make_response(render_template('swagger-ui.html', title=api.title, specs_url=specs_url), 200)


@api.route('/api/server-status')
class ServerStatus(Resource):
    @staticmethod
    def get():
        return {}, 200

class PredictResource(Resource):
    @api.expect(carprice_model)
    def post(self):
        try:
            
            search_json = json.loads(request.data.decode('utf-8'))
            year = search_json["year"]
            present_price = search_json["present_price"]
            owner = search_json["owner"]
            kms_driven = search_json["kms_driven"]
            fuel_type = search_json["fuel_type"]
            seller_type = search_json["seller_type"]
            transmission_type = search_json["transmission_type"]
            
            results = prediction.make_predictions(year=year, present_price=present_price, kms_driven=kms_driven, owner=owner,
            fuel_type=fuel_type, seller_type=seller_type, transmission_type=transmission_type)            
            return results

        except Exception as e:
            return e


mp_api.add_resource(PredictResource, '/predict', methods=['POST'])


if __name__ == '__main__':
    app.run(debug=True , port=5006, host="127.0.0.1")












    