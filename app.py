from flask import Flask, jsonify, request, make_response
from neuralNetworkBackEnd import NeuralNetwork
import neuralTester
import numpy as np
import csv
import json
from flask_cors import CORS
import pandas as pd
from io import StringIO
app = Flask(__name__)
CORS(app)



#ruta
@app.route('/prediction', methods=['POST'])
def prediction():
    fixedAcidity = float(request.json['fixedAcidity'])
    volatileAcidity =float(request.json['volatileAcidity'])
    citricAcid = float(request.json['citricAcid'])
    residualSugar =float(request.json['residualSugar'])
    chlorides = float(request.json['chlorides'])
    freeSulfurDioxide = float(request.json['freeSulfurDioxide'])
    totalSulfurDioxide = float(request.json['totalSulfurDioxide'])
    density = float(request.json['density'])
    pH = float(request.json['pH'])
    sulphates = float(request.json['sulphates'])
    alcohol = float(request.json['alcohol'])
    nn = NeuralNetwork()
    nn.train()
    quality = float(nn.predict(fixedAcidity, volatileAcidity, citricAcid, residualSugar, chlorides, freeSulfurDioxide, totalSulfurDioxide, density, pH, sulphates, alcohol))
    return jsonify({"quality":quality})

@app.route('/dataset')
def dataset():
    csv = (pd.read_csv('dataSet.csv', sep=";"))
    return jsonify({"csv":csv.to_json()})

if __name__ == '__main__':
    app.run(debug=True, port=4000)

