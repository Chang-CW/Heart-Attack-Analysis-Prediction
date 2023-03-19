from flask import Flask, request, jsonify
import pickle
import numpy as np
# from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#socketio = SocketIO(app, cors_allowed_origins='*')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 16)
    loaded_model = pickle.load(open("pkl/heart_KNN.pkl", "rb"))
    result = loaded_model.predict_proba(to_predict)
    return round(result[0][1]*100, 2)

@app.route('/api', methods = ['GET'])
def returnProb():
    # d = {}
    # inputchr = str(request.args['query'])
    # answer = str(ord(inputchr))
    # d['output'] = answer
    # return d
    d = {}
    X = ['age', 'sex', 'cp', 'trtbps', 'chol',
         'fbs', 'restecg', 'thalachh', 'exng',
         'oldpeak', 'slp', 'caa', 'thall']
    to_predict_list = []
    for x in X:
        to_predict_list.append(int(request.args[x]))
    print(to_predict_list)
    # to_predict_list.append(int(request.args['Age']))    
    # to_predict_list.append(int(request.args['Gender']))
    # to_predict_list.append(int(request.args['Polyuria']))
    # to_predict_list.append(int(request.args['Polydipsia']))
    # to_predict_list.append(int(request.args['sudden_weight_loss']))
    # to_predict_list.append(int(request.args['weakness']))
    # to_predict_list.append(int(request.args['Polyphagia']))
    # to_predict_list.append(int(request.args['Genital_thrush']))
    # to_predict_list.append(int(request.args['visual_blurring']))
    # to_predict_list.append(int(request.args['Itching']))
    # to_predict_list.append(int(request.args['Irritability']))
    # to_predict_list.append(int(request.args['delayed_healing']))
    # to_predict_list.append(int(request.args['partial_paresis']))
    # to_predict_list.append(int(request.args['muscle_stiffness']))
    # to_predict_list.append(int(request.args['Alopecia']))
    # to_predict_list.append(int(request.args['Obesity']))
    d['output'] = str(ValuePredictor(to_predict_list))
    # return str(to_predict_list)
    return d

if __name__ =="__main__":
    app.run(debug=True)
