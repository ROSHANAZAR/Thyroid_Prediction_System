from Thyroid import *
from flask import Flask,render_template,request,jsonify
from flask_cors import CORS
'''print(find_accuracy(1))
print(predict("Random Forest",0,X_new = [[75, 0, 1.6, 0.89, 0.05]]))
print(final_prediction(1,X_new = [[23, 0, 2.8, 1.14, 0.1]]))'''

app = Flask(__name__)

"CORS(app)"

@app.route('/')
def index():
    return render_template("Thyroid.html")

@app.route('/Accuracy',methods=['POST','GET'])
def accuracy():
    if request.method=="GET":
        Dec_Avg,Dec_max,Ran_Avg,Ran_max,Log_Avg,Log_max,Naive_Avg,Naive_max=find_accuracy(0)

        print(Dec_Avg,Dec_max,Ran_Avg,Ran_max,Log_Avg,Log_max,Naive_Avg,Naive_max)

        Accuracy={"Dec_Avg":Dec_Avg,"Dec_max":Dec_max,"Ran_Avg":Ran_Avg,"Ran_max":Ran_max,
                  "Log_Avg":Log_Avg,"Log_max":Log_max,"Naive_Avg":Naive_Avg,"Naive_max":Naive_max}

        return jsonify(Accuracy)

@app.route('/Predict', methods=['POST'])
def post_example():
    data = request.get_json()
    name = data['name']
    age = int(data['age'])
    gender = data['gender']
    pregnancy=data['pregnancy']
    t3 = float(data['t3'])
    t4 = float(data['t4'])
    tsh = float(data['tsh'])

    print(data)

    if pregnancy=="pregnant":
        pregnancy=1
    else:
        pregnancy=0

    if gender=="male":
        gender=1
    else:
        gender=0

    status=final_prediction(pregnancy,[[age,gender,t3,t4,tsh]])

    print(status)

    Dec_Avg, Dec_max, Ran_Avg, Ran_max, Log_Avg, Log_max, Naive_Avg, Naive_max = find_accuracy(pregnancy)

    print(Dec_Avg, Dec_max, Ran_Avg, Ran_max, Log_Avg, Log_max, Naive_Avg, Naive_max)

    Dictionary = {"Dec_Avg": Dec_Avg, "Dec_max": Dec_max, "Ran_Avg": Ran_Avg, "Ran_max": Ran_max,
                "Log_Avg": Log_Avg, "Log_max": Log_max, "Naive_Avg": Naive_Avg, "Naive_max": Naive_max,
                  'status':status[0]}

    return jsonify(Dictionary)








if __name__ == '__main__':
    app.run(debug=True)







