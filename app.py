from flask import Flask, jsonify, request
from neuralNetworkBackEnd import NeuralNetwork
import neuralTester
import numpy as np
from flask_cors import CORS
import pandas as pd

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
    csv = pd.read_csv('dataSet.csv', sep=";")
    return jsonify({"csv":csv})

if __name__ == '__main__':
    app.run(debug=True, port=4000)

