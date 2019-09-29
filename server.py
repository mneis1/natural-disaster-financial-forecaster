from flask import Flask, request
import pandas as pd
import json
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

    dict_list = []
    split_preds = np.split(predictions[0],50)
    for preds in split_preds:
        preds_dict = {i:preds[i] for i in range(0, len(preds))}
        mod_preds_dict = {1:preds_dict[0], 2:preds_dict[1], 3:preds_dict[2], 4:preds_dict[3]}
        dict_list.append(mod_preds_dict)

    response = {
    'alabama':dict_list[0],
    'alaska':dict_list[1],
    'arkansas':dict_list[2],
    'arizona':dict_list[3],
    'california':dict_list[4],
    'colorado':dict_list[5],
    'connecticut':dict_list[6],
    'delaware':dict_list[7],
    'florida':dict_list[8],
    'georgia':dict_list[9],
    'hawaii':dict_list[10],
    'iowa':dict_list[11],
    'idaho':dict_list[12],
    'illinois':dict_list[13],
    'indiana':dict_list[14],
    'kansas':dict_list[15],
    'kentucky':dict_list[16],
    'louisiana':dict_list[17],
    'massachusetts':dict_list[18],
    'maryland':dict_list[19],
    'maine':dict_list[20],
    'michigan':dict_list[21],
    'minnesota':dict_list[22],
    'missouri':dict_list[23],
    'mississippi':dict_list[24],
    'montana':dict_list[25],
    'north carolina':dict_list[26],
    'north dakota':dict_list[27],
    'nebraska':dict_list[28],
    'new hampshire':dict_list[29],
    'new jersey':dict_list[30],
    'new mexico':dict_list[31],
    'nevada':dict_list[32],
    'new york':dict_list[33],
    'ohio':dict_list[34],
    'oklahoma':dict_list[35],
    'oregon':dict_list[36],
    'pennsylvania':dict_list[37],
    'rhode island':dict_list[38],
    'south carolina':dict_list[39],
    'south dakota':dict_list[40],
    'tennessee':dict_list[41],
    'texas':dict_list[42],
    'utah':dict_list[43],
    'virginia':dict_list[44],
    'vermont':dict_list[45],
    'washington':dict_list[46],
    'wisconsin':dict_list[47],
    'west virginia':dict_list[48],
    'wyoming':dict_list[49]
    }

    return response
    #return "{\"alabama\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"alaska\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"arizona\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"arkansas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"california\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"colorado\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"connecticut\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"delaware\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"florida\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"georgia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"hawaii\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"idaho\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"illinois\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"indiana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"iowa\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"kansas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"kentucky\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"louisiana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"maine\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"maryland\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"massachusetts\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"michigan\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"minnesota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"mississippi\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"missouri\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"montana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"nebraska\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"nevada\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new hampshire\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new jersey\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new mexico\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new york\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"north carolina\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"north dakota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"ohio\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"oklahoma\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"oregon\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"pennsylvania\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"rhode island\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"south carolina\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"south dakota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"tennessee\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"texas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"utah\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"vermont\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"virginia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"washington\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"west virginia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"wisconsin\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"wyoming\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5}}"


@app.route("/getdisasters", methods=['POST', 'GET'])
def disasterdata():
    df = pd.read_csv("historic-full-data.csv")
    arrofallobjs = []
    for index, row in df.iterrows():
        # declarationDate	Year	Quarter	state	incidentType	alabama-Q1	alaska-Q1 ...
        #print(row[index], row[index+1])
        innerDictElement = {'year': row[1], 'quarter': row[2], 'state':row[3], 'disaster':row[4]}

        # gdpDict = dict()
        # for i in range(50):
        #     gdpDict[df.columns[i + 5].split(' ')[0]] = row[i+5]
        #
        # innerDictElement["gdp"] = gdpDict
        arrofallobjs.append(innerDictElement)
    output_json = json.dumps(arrofallobjs)
    print(str(output_json))

    return str(output_json)

@app.route("/gethistoric", methods=['POST', 'GET'])
def gethistoricdata():
    mainDict = dict()
    df = pd.read_csv("GDPByState.csv")
    for index, row in df.iterrows():

        tempDict = dict()
        for i in range(51):
            tempDict[str(i)] = row[i+1]
        mainDict[row[0]] = tempDict

    output_json = json.dumps(mainDict)
    print(str(output_json))

    return str(output_json)

if __name__== '__main__':
    app.run(host='0.0.0.0')
