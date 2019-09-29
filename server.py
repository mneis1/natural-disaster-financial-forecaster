from flask import Flask, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np


app = Flask(__name__)

@app.route("/")
def mainPage():
    return "Use the /pulldata endpoint to retrive the most recently trained predictions."


@app.route("/pulldata", methods=['POST'])
def pullData():

    data = request.form
    #TODO
    disaster_to_int = {
    'Tornado':0, 
    'Flood':1, 
    'Fire':2, 
    'Earthquake':3, 
    'Hurricane':4,
    'Volcano':5, 
    'Severe Storm(s)':6, 
    'Typhoon':7,
    'Drought':8, 
    'Snow':9, 
    'Severe Ice Storm':10,
    'Freezing':11, 
    'Coastal Storm':12, 
    'Mud/Landslide':13,
    'Tsunami':14
    }

    states_vals = {
    'alabama':0,
    'alaska':1,
    'arkansas':2,
    'arizona':3,
    'california':4,
    'colorado':5,
    'connecticut':6,
    'delaware':7,
    'florida':8,
    'georgia':9,
    'hawaii':10,
    'iowa':11,
    'idaho':12,
    'illinois':13,
    'indiana':14,
    'kansas':15,
    'kentucky':16,
    'louisiana':17,
    'massachusetts':18,
    'maryland':19,
    'maine':20,
    'michigan':21,
    'minnesota':22,
    'missouri':23,
    'mississippi':24,
    'montana':25,
    'north carolina':26,
    'north dakota':27,
    'nebraska':28,
    'new hampshire':29,
    'new jersey':30,
    'new mexico':31,
    'nevada':32,
    'new york':33,
    'ohio':34,
    'oklahoma':35,
    'oregon':36,
    'pennsylvania':37,
    'rhode island':38,
    'south carolina':39,
    'south dakota':40,
    'tennessee':41,
    'texas':42,
    'utah':43,
    'virginia':44,
    'vermont':45,
    'washington':46,
    'wisconsin':47,
    'west virginia':48,
    'wyoming':49
    }

    state = pd.Series([data['state']]).map(states_vals)
    disaster = pd.Series([data['disaster']]).map(disaster_to_int)
    d = [[state.values[0], disaster.values[0]]]

    random_forest = joblib.load('natural-disaster-financial-forecaster/random_f.m')
    predictions = random_forest.predict(d)
    #d_tree = joblib.load('natural-disaster-financial-forecaster/des_tree.sav')
    #predictions = d_tree.predict(d)

    split_preds = np.split(predictions[0],50)

    response = {
    'alabama':split_preds[0].tolist(),
    'alaska':split_preds[1].tolist(),
    'arkansas':split_preds[2].tolist(),
    'arizona':split_preds[3].tolist(),
    'california':split_preds[4].tolist(),
    'colorado':split_preds[5].tolist(),
    'connecticut':split_preds[6].tolist(),
    'delaware':split_preds[7].tolist(),
    'florida':split_preds[8].tolist(),
    'georgia':split_preds[9].tolist(),
    'hawaii':split_preds[10].tolist(),
    'iowa':split_preds[11].tolist(),
    'idaho':split_preds[12].tolist(),
    'illinois':split_preds[13].tolist(),
    'indiana':split_preds[14].tolist(),
    'kansas':split_preds[15].tolist(),
    'kentucky':split_preds[16].tolist(),
    'louisiana':split_preds[17].tolist(),
    'massachusetts':split_preds[18].tolist(),
    'maryland':split_preds[19].tolist(),
    'maine':split_preds[20].tolist(),
    'michigan':split_preds[21].tolist(),
    'minnesota':split_preds[22].tolist(),
    'missouri':split_preds[23].tolist(),
    'mississippi':split_preds[24].tolist(),
    'montana':split_preds[25].tolist(),
    'north carolina':split_preds[26].tolist(),
    'north dakota':split_preds[27].tolist(),
    'nebraska':split_preds[28].tolist(),
    'new hampshire':split_preds[29].tolist(),
    'new jersey':split_preds[30].tolist(),
    'new mexico':split_preds[31].tolist(),
    'nevada':split_preds[32].tolist(),
    'new york':split_preds[33].tolist(),
    'ohio':split_preds[34].tolist(),
    'oklahoma':split_preds[35].tolist(),
    'oregon':split_preds[36].tolist(),
    'pennsylvania':split_preds[37].tolist(),
    'rhode island':split_preds[38].tolist(),
    'south carolina':split_preds[39].tolist(),
    'south dakota':split_preds[40].tolist(),
    'tennessee':split_preds[41].tolist(),
    'texas':split_preds[42].tolist(),
    'utah':split_preds[43].tolist(),
    'virginia':split_preds[44].tolist(),
    'vermont':split_preds[45].tolist(),
    'washington':split_preds[46].tolist(),
    'wisconsin':split_preds[47].tolist(),
    'west virginia':split_preds[48].tolist(),
    'wyoming':split_preds[49].tolist()
    }

    return response
    #return "{\"alabama\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"alaska\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"arizona\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"arkansas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"california\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"colorado\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"connecticut\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"delaware\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"florida\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"georgia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"hawaii\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"idaho\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"illinois\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"indiana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"iowa\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"kansas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"kentucky\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"louisiana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"maine\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"maryland\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"massachusetts\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"michigan\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"minnesota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"mississippi\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"missouri\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"montana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"nebraska\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"nevada\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new hampshire\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new jersey\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new mexico\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new york\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"north carolina\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"north dakota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"ohio\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"oklahoma\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"oregon\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"pennsylvania\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"rhode island\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"south carolina\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"south dakota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"tennessee\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"texas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"utah\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"vermont\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"virginia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"washington\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"west virginia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"wisconsin\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"wyoming\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5}}"


@app.route("/gethistoric", methods=['POST', 'GET'])
def gethistoricdata():
    df = pd.read_csv("historic-full-data.csv")
    for index, row in df.iterrows():
        print("index is " + str(index))
        print("row is " + str(row))
        break

    return ""

if __name__== '__main__':
    app.run(host='0.0.0.0')
