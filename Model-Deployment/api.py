from flask import Flask, request, render_template,jsonify, send_file
import joblib 
import pandas as pd 

# FLASK APP
app = Flask(__name__)

# HTML FILE
@app.route('/')
def home():
    return render_template('index.html')

# API CALL
@app.route('/predict', methods=['POST'])
def predict():
    
    # JSON REQUEST
    feature_data = request.json
    
    # CONVERT JSON -> PANDAS DF
    df = pd.DataFrame(feature_data)
    df = df.reindex(columns=col_names)
    # PREDICT
    prediction = model.predict(df)[0]
    
    formatted_prediction = round(float(prediction), 2)
    return jsonify({'prediction': str(formatted_prediction)})

if __name__ == '__main__':
    model = joblib.load('final_model.pkl')
    col_names = joblib.load('col_names.pkl')
    
    app.run(debug=True)