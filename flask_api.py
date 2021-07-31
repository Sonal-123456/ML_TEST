from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("C:/Users/sonal/Documents/M-DL Projects/1.Stock-Sentiment-Analysis/classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
  return "Welcome All"

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Analize stock sentiments  
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))



if __name__=='__main__':
 app.run(host='0.0.0.0',port=8000)



    
    