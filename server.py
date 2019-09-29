from flask import Flask
import pandas as pd
import json

app = Flask(__name__)

@app.route("/")
def mainPage():
    return "Use the /pulldata endpoint to retrive the most recently trained predictions."


@app.route("/pulldata", methods=['POST'])
def pullData():
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
    return "{\"alabama\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"alaska\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"arizona\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"arkansas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"california\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"colorado\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"connecticut\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"delaware\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"florida\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"georgia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"hawaii\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"idaho\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"illinois\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"indiana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"iowa\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"kansas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"kentucky\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"louisiana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"maine\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"maryland\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"massachusetts\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"michigan\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"minnesota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"mississippi\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"missouri\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"montana\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"nebraska\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"nevada\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new hampshire\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new jersey\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new mexico\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"new york\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"north carolina\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"north dakota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"ohio\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"oklahoma\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"oregon\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"pennsylvania\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"rhode island\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"south carolina\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"south dakota\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"tennessee\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"texas\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"utah\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"vermont\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"virginia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"washington\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"west virginia\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"wisconsin\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5},\"wyoming\":{\"1\":0.5,\"2\":0.5,\"3\":0.5,\"4\":0.5}}"


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
